#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <map>

struct Bar {
    std::chrono::system_clock::time_point ts;
    double open{};
    double high{};
    double low{};
    double close{};
    double atr{}; // optional
};

struct Signal {
    std::chrono::system_clock::time_point entry_time;
    std::string direction; // "UP" oder "DOWN"
    double stop_loss{};
    double take_profit{};
    double confidence{}; // optional
    std::string entry_tf;
    std::string setup;
};

struct SymbolInfo {
    double trade_tick_size{0.01};
    double trade_tick_value{1.0};
    double trade_contract_size{100.0};
    double point{0.01};
    double volume_step{0.01};
    double volume_min{0.01};
    double volume_max{100.0};
    double leverage{30.0};
    double commission_round_per_lot{7.0};
};

struct Config {
    double account_balance{10'000.0};
    double risk_per_trade{0.04};
    double risk_per_trade_min{0.04};
    double risk_per_trade_max{0.04};
    double max_gross_exposure_pct{0.05};
    bool size_by_prob{false};
    double prob_size_min{0.7};
    double prob_size_max{1.5};
    double ml_probability_threshold{0.65};
    double size_short_factor{0.7};
    double min_lot{0.01};
    double max_lot{100.0};
    double trailing_atr_mult{0.0};
    bool sort_inputs{true};
    bool stop_hits_before_tp{true};
    bool sharpe_daily{true};
    bool pct_return_scale_100{true};
    bool use_prob_risk_scaling{true};
    bool use_prob_size_scaling{true};
};

struct TradeResult {
    std::chrono::system_clock::time_point entry_time;
    std::chrono::system_clock::time_point exit_time;
    std::string direction;
    double entry{};
    double exit{};
    double stop{};
    double tp{};
    double lots{};
    double pnl{};
    double pnl_pct{};
    double ret{}; // per-trade return (fraction of entry equity)
    double risk_amount{};
    double rr{};
    int bars_held{};
};

static std::chrono::system_clock::time_point parse_time(const std::string &s) {
    // Accept epoch milliseconds/seconds or formatted timestamps
    if (!s.empty() && std::all_of(s.begin(), s.end(), ::isdigit)) {
        long long epoch = std::stoll(s);
        // Heuristic: >1e12 => milliseconds
        if (epoch > 1'000'000'000'000LL) {
            return std::chrono::system_clock::time_point{std::chrono::milliseconds(epoch)};
        }
        return std::chrono::system_clock::time_point{std::chrono::seconds(epoch)};
    }
    std::tm tm{};
    std::istringstream ss(s);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (ss.fail()) {
        ss.clear();
        ss.str(s);
        ss >> std::get_time(&tm, "%Y-%m-%d");
    }
    auto tt = std::mktime(&tm);
    return std::chrono::system_clock::from_time_t(tt);
}

static std::vector<Bar> load_bars(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open bars file: " + path);
    std::vector<Bar> out;
    std::string line;
    // header
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        Bar b;
        std::getline(ss, cell, ',');
        b.ts = parse_time(cell);
        std::getline(ss, cell, ',');
        b.open = std::stod(cell);
        std::getline(ss, cell, ',');
        b.high = std::stod(cell);
        std::getline(ss, cell, ',');
        b.low = std::stod(cell);
        std::getline(ss, cell, ',');
        b.close = std::stod(cell);
        if (std::getline(ss, cell, ',')) {
            if (!cell.empty()) b.atr = std::stod(cell);
        }
        out.push_back(b);
    }
    std::sort(out.begin(), out.end(), [](const Bar &a, const Bar &b) { return a.ts < b.ts; });
    return out;
}

static std::vector<Signal> load_signals(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open signals file: " + path);
    std::vector<Signal> out;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        Signal s;
        std::getline(ss, cell, ',');
        s.entry_time = parse_time(cell);
        std::getline(ss, cell, ',');
        s.direction = cell;
        std::getline(ss, cell, ',');
        s.stop_loss = std::stod(cell);
        std::getline(ss, cell, ',');
        s.take_profit = std::stod(cell);
        if (std::getline(ss, cell, ',')) s.confidence = cell.empty() ? 0.0 : std::stod(cell);
        out.push_back(s);
    }
    std::sort(out.begin(), out.end(), [](const Signal &a, const Signal &b) { return a.entry_time < b.entry_time; });
    return out;
}

class Backtester {
  public:
    Backtester(Config cfg, SymbolInfo info) : cfg_(cfg), info_(info), equity_(cfg.account_balance), high_equity_(equity_) {}

    void run(const std::vector<Bar> &bars, const std::vector<Signal> &signals) {
        bars_ = &bars;
        for (const auto &sig : signals) {
            auto res = simulate_trade(sig);
            if (res) trades_.push_back(*res);
        }
    }

    const std::vector<TradeResult> &trades() const { return trades_; }

    std::unordered_map<std::string, double> summary() const {
        std::unordered_map<std::string, double> m;
        if (trades_.empty()) return m;
        double start_equity = cfg_.account_balance;
        double equity = start_equity;
        std::vector<std::pair<std::chrono::system_clock::time_point, double>> eq;
        std::vector<TradeResult> ordered = trades_;
        std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) { return a.exit_time < b.exit_time; });
        for (auto &t : ordered) {
            equity += t.pnl;
            eq.push_back({t.exit_time, equity});
        }
        double max_dd_pct = 0.0, max_dd_abs = 0.0;
        {
            double peak = start_equity;
            for (auto &p : eq) {
                if (p.second > peak) peak = p.second;
                double dd_abs = peak - p.second;
                double dd_pct = (peak - p.second) / peak * 100.0;
                if (dd_abs > max_dd_abs) max_dd_abs = dd_abs;
                if (dd_pct > max_dd_pct) max_dd_pct = dd_pct;
            }
            max_dd_pct = -max_dd_pct;
            max_dd_abs = -max_dd_abs;
        }
        std::vector<double> pnls;
        pnls.reserve(trades_.size());
        std::vector<double> trade_returns;
        trade_returns.reserve(trades_.size());
        for (auto &t : trades_) {
            pnls.push_back(t.pnl);
            trade_returns.push_back(t.ret);
        }
        int wins = std::count_if(trades_.begin(), trades_.end(), [](const auto &t) { return t.pnl > 0; });
        int losses = trades_.size() - wins;
        double sum_win = 0.0, sum_loss = 0.0;
        for (auto &t : trades_) {
            if (t.pnl > 0)
                sum_win += t.pnl;
            else
                sum_loss += t.pnl;
        }
        double profit_factor = (losses == 0) ? std::numeric_limits<double>::infinity() : sum_win / std::abs(sum_loss);
        std::vector<double> rr_vec;
        rr_vec.reserve(trades_.size());
        for (auto &t : trades_) rr_vec.push_back(t.rr);
        auto mean = [](const std::vector<double> &v) {
            if (v.empty()) return 0.0;
            return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };
        auto median = [](std::vector<double> v) {
            if (v.empty()) return 0.0;
            size_t n = v.size();
            std::nth_element(v.begin(), v.begin() + n / 2, v.end());
            double med = v[n / 2];
            if (n % 2 == 0) {
                std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
                med = 0.5 * (med + v[n / 2 - 1]);
            }
            return med;
        };
        m["trades"] = static_cast<double>(trades_.size());
        m["winrate_pct"] = trades_.empty() ? 0.0 : (wins * 100.0 / trades_.size());
        m["pnl_abs"] = equity - start_equity;
        m["total_return_pct"] = (equity / start_equity - 1.0) * 100.0;
        m["profit_factor"] = profit_factor;
        m["avg_pnl"] = mean(pnls);
        m["median_pnl"] = median(pnls);
        m["avg_rr"] = mean(rr_vec);
        std::vector<double> hold;
        hold.reserve(trades_.size());
        for (auto &t : trades_) hold.push_back(static_cast<double>(t.bars_held));
        m["avg_hold_bars"] = mean(hold);
        m["max_dd_pct"] = max_dd_pct;
        m["max_dd_abs"] = max_dd_abs;
        m["equity_end"] = equity;
        // Sharpe per trade (using fractional returns)
        if (trade_returns.size() > 1) {
            double mean_r = mean(trade_returns);
            double var = 0.0;
            for (auto r : trade_returns) var += (r - mean_r) * (r - mean_r);
            var /= trade_returns.size();
            double stddev = std::sqrt(var);
            m["sharpe_trade"] = (stddev > 0) ? (mean_r / stddev * std::sqrt(static_cast<double>(trade_returns.size()))) : 0.0;
        } else {
            m["sharpe_trade"] = 0.0;
        }
        // Sharpe (daily): carry forward equity for every calendar day between first/last trade
        std::vector<double> daily_returns;
        if (cfg_.sharpe_daily && eq.size() > 1) {
            std::map<long long, double> daily_last; // day -> last equity that day
            for (auto &p : eq) {
                auto hours = std::chrono::duration_cast<std::chrono::hours>(p.first.time_since_epoch()).count();
                long long day_bucket = hours / 24;
                daily_last[day_bucket] = p.second; // overwrite keeps last-in-day
            }
            long long first_day = daily_last.begin()->first;
            long long last_day = daily_last.rbegin()->first;
            double prev_eq = start_equity; // equity before first trading day
            for (long long d = first_day; d <= last_day; ++d) {
                double cur_eq = prev_eq;
                auto it = daily_last.find(d);
                if (it != daily_last.end()) cur_eq = it->second;
                double r = (cur_eq / prev_eq) - 1.0;
                daily_returns.push_back(r);
                prev_eq = cur_eq;
            }
        }
        if (!daily_returns.empty()) {
            double mean_r = mean(daily_returns);
            double var = 0.0;
            for (auto r : daily_returns) var += (r - mean_r) * (r - mean_r);
            var /= daily_returns.size();
            double stddev = std::sqrt(var);
            m["sharpe_daily"] = (stddev > 0) ? (mean_r / stddev * std::sqrt(252.0)) : 0.0;
        } else {
            m["sharpe_daily"] = 0.0;
        }
        // CAGR
        auto duration_days = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<86400>>>(eq.back().first - eq.front().first).count();
        double years = duration_days / 365.25;
        if (years > 0.0 && equity > 0.0 && start_equity > 0.0) {
            m["cagr"] = std::pow(equity / start_equity, 1.0 / years) - 1.0;
        } else {
            m["cagr"] = 0.0;
        }
        return m;
    }

  private:
    std::optional<TradeResult> simulate_trade(const Signal &sig) {
        // locate entry index
        const auto &bars = *bars_;
        auto it = std::lower_bound(bars.begin(), bars.end(), sig.entry_time,
                                   [](const Bar &b, const std::chrono::system_clock::time_point &t) { return b.ts < t; });
        if (it == bars.end()) return std::nullopt;
        size_t idx = static_cast<size_t>(std::distance(bars.begin(), it));
        const auto &entry_bar = bars[idx];
        double entry_price = entry_bar.close;
        double stop_price = sig.stop_loss;
        double tp_price = sig.take_profit;

        double lots, risk_amount, stop_dist;
        std::tie(lots, risk_amount, stop_dist) = calculate_volume(sig, stop_price, entry_price);
        if (lots <= 0.0) return std::nullopt;

        double entry_equity = equity_;

        double best = entry_price;
        double trailing = stop_price;
        int bars_held = 0;
        double exit_price = entry_price;
        auto exit_time = entry_bar.ts;
        const bool is_long = (sig.direction == "UP");

        bool hit_level = false;
        for (size_t i = idx + 1; i < bars.size(); ++i) {
            const auto &b = bars[i];
            double high = b.high;
            double low = b.low;
            double atr = b.atr;
            ++bars_held;
            if (is_long) {
                best = std::max(best, high);
                if (cfg_.trailing_atr_mult > 0.0 && atr > 0.0) trailing = std::max(trailing, best - atr * cfg_.trailing_atr_mult);
                double stop_level = trailing;
                if (cfg_.stop_hits_before_tp && low <= stop_level) { exit_price = stop_level; exit_time = b.ts; hit_level = true; break; }
                if (high >= tp_price) { exit_price = tp_price; exit_time = b.ts; hit_level = true; break; }
            } else {
                best = std::min(best, low);
                if (cfg_.trailing_atr_mult > 0.0 && atr > 0.0) trailing = std::min(trailing, best + atr * cfg_.trailing_atr_mult);
                double stop_level = trailing;
                if (cfg_.stop_hits_before_tp && high >= stop_level) { exit_price = stop_level; exit_time = b.ts; hit_level = true; break; }
                if (low <= tp_price) { exit_price = tp_price; exit_time = b.ts; hit_level = true; break; }
            }
            exit_time = b.ts;
            exit_price = b.close;
        }

        double direction_sign = is_long ? 1.0 : -1.0;
        double tick_size = info_.trade_tick_size > 0 ? info_.trade_tick_size : info_.point;
        double tick_value = info_.trade_tick_value > 0 ? info_.trade_tick_value : info_.trade_contract_size;
        double ticks = (exit_price - entry_price) / tick_size;
        double pnl = ticks * tick_value * direction_sign * lots;
        double commission = info_.commission_round_per_lot * lots;
        pnl -= commission;
        double pnl_pct = (pnl / std::max(1e-9, equity_)) * (cfg_.pct_return_scale_100 ? 100.0 : 1.0);
        double rr = (std::fabs(exit_price - entry_price) / std::max(stop_dist, 1e-9)) * direction_sign;
        double ret = pnl / std::max(1e-9, entry_equity);

        equity_ += pnl;
        if (equity_ > high_equity_) high_equity_ = equity_;

        TradeResult tr{entry_bar.ts, exit_time, sig.direction, entry_price, exit_price, stop_price, tp_price, lots,
                       pnl, pnl_pct, ret, risk_amount, rr, bars_held};
        return tr;
    }

    std::tuple<double, double, double> calculate_volume(const Signal &sig, double stop_price, double entry_price) const {
        double stop_distance = std::fabs(entry_price - stop_price);
        double balance = equity_;
        double risk_fraction = std::clamp(cfg_.risk_per_trade, cfg_.risk_per_trade_min, cfg_.risk_per_trade_max);

        auto prob_scale = [&](double prob) {
            double base_thr = std::max(0.5, cfg_.ml_probability_threshold);
            double frac = std::clamp((prob - base_thr) / std::max(1e-6, 1.0 - base_thr), 0.0, 1.0);
            return cfg_.prob_size_min + (cfg_.prob_size_max - cfg_.prob_size_min) * frac;
        };

        double risk_amount = risk_fraction * balance;
        if (cfg_.size_by_prob && cfg_.use_prob_risk_scaling) {
            risk_amount *= prob_scale(sig.confidence);
        }
        if (stop_distance <= 0.0 || risk_amount <= 0.0) return {cfg_.min_lot, risk_amount, stop_distance};

        double tick_size = info_.trade_tick_size > 0 ? info_.trade_tick_size : info_.point;
        double tick_value = info_.trade_tick_value > 0 ? info_.trade_tick_value : info_.trade_contract_size;
        double ticks = stop_distance / tick_size;
        double risk_per_lot = ticks * tick_value;
        if (risk_per_lot <= 0) risk_per_lot = stop_distance * info_.trade_contract_size;

        double lots = risk_amount / risk_per_lot;
        if (cfg_.size_by_prob && cfg_.use_prob_size_scaling) {
            lots *= prob_scale(sig.confidence);
        }
        if (sig.direction == "DOWN") lots *= cfg_.size_short_factor;
        lots = std::clamp(lots, cfg_.min_lot, cfg_.max_lot);
        lots = cap_to_exposure(balance, entry_price, lots);
        double realized_risk_amount = lots * risk_per_lot;
        return {lots, realized_risk_amount, stop_distance};
    }

    double exposure_value(double price, double lots) const {
        return price * info_.trade_contract_size * lots;
    }

    double cap_to_exposure(double balance, double entry_price, double lots) const {
        if (lots <= 0) return lots;
        double notional = exposure_value(entry_price, lots);
        double limit_pct = std::max(cfg_.max_gross_exposure_pct, 0.0);
        if (limit_pct > 0) {
            double remaining = balance * limit_pct;
            if (notional > remaining) {
                double factor = remaining / std::max(1e-9, notional);
                lots = std::max(cfg_.min_lot, lots * factor);
            }
        }
        if (info_.leverage > 0) {
            double margin_per_lot = (entry_price * info_.trade_contract_size) / info_.leverage;
            double max_lots_margin = balance / std::max(1e-9, margin_per_lot);
            lots = std::min(lots, max_lots_margin);
        }
        lots = std::clamp(lots, cfg_.min_lot, info_.volume_max);
        double step = std::max(info_.volume_step, 1e-9);
        double steps = std::floor(lots / step);
        lots = steps * step;
        return std::max(cfg_.min_lot, std::min(lots, info_.volume_max));
    }

    Config cfg_;
    SymbolInfo info_;
    double equity_;
    double high_equity_;
    const std::vector<Bar> *bars_{};
    std::vector<TradeResult> trades_;
};

static void print_summary(const std::unordered_map<std::string, double> &m) {
    for (const auto &kv : m) {
        std::cout << kv.first << ": " << kv.second << "\n";
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: backtester <bars_csv> <signals_csv>\n";
        return 1;
    }
    std::string bars_path = argv[1];
    std::string signals_path = argv[2];

    try {
        auto bars = load_bars(bars_path);
        auto signals = load_signals(signals_path);
        Config cfg; // defaults as in Python backtester
        SymbolInfo info; // ICMarkets style defaults
        Backtester bt(cfg, info);
        bt.run(bars, signals);
        auto summary = bt.summary();
        std::cout << "Trades: " << bt.trades().size() << "\n";
        print_summary(summary);
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
