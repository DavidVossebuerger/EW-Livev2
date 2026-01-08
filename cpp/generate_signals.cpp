#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>

struct Bar {
    std::chrono::system_clock::time_point ts;
    double open{};
    double high{};
    double low{};
    double close{};
};

static std::chrono::system_clock::time_point parse_time(const std::string &s) {
    if (!s.empty() && std::all_of(s.begin(), s.end(), ::isdigit)) {
        long long epoch = std::stoll(s);
        if (epoch > 1'000'000'000'000LL) return std::chrono::system_clock::time_point{std::chrono::milliseconds(epoch)};
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
    std::getline(f, line); // header
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
        out.push_back(b);
    }
    std::sort(out.begin(), out.end(), [](const Bar &a, const Bar &b) { return a.ts < b.ts; });
    return out;
}

struct WindowHL {
    size_t period;
    std::deque<std::pair<size_t, double>> highs;
    std::deque<std::pair<size_t, double>> lows;
    explicit WindowHL(size_t p) : period(p) {}
    void push(size_t idx, double high, double low) {
        while (!highs.empty() && highs.back().second <= high) highs.pop_back();
        while (!lows.empty() && lows.back().second >= low) lows.pop_back();
        highs.emplace_back(idx, high);
        lows.emplace_back(idx, low);
    }
    void evict(size_t idx_floor) {
        while (!highs.empty() && highs.front().first < idx_floor) highs.pop_front();
        while (!lows.empty() && lows.front().first < idx_floor) lows.pop_front();
    }
    double max() const { return highs.empty() ? std::numeric_limits<double>::quiet_NaN() : highs.front().second; }
    double min() const { return lows.empty() ? std::numeric_limits<double>::quiet_NaN() : lows.front().second; }
};

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: generate_signals <bars_csv> <output_csv> <lookback> [tp_mult=2.0] [sl_buffer=0.0]\n";
        return 1;
    }
    std::string bars_path = argv[1];
    std::string out_path = argv[2];
    size_t lookback = static_cast<size_t>(std::stoul(argv[3]));
    double tp_mult = (argc >= 5) ? std::stod(argv[4]) : 2.0;
    double sl_buffer = (argc >= 6) ? std::stod(argv[5]) : 0.0;

    try {
        auto bars = load_bars(bars_path);
        WindowHL win(lookback);
        std::ofstream out(out_path);
        out << "entry_time,direction,stop_loss,take_profit,confidence\n";
        for (size_t i = 0; i < bars.size(); ++i) {
            const auto &b = bars[i];
            if (i > 0) {
                const auto &prev = bars[i - 1];
                win.evict((i - 1) >= lookback ? (i - 1 - lookback) : 0);
                win.push(i - 1, prev.high, prev.low);
            }
            if (i < lookback + 1) continue; // need full window behind current bar

            double recent_high = win.max();
            double recent_low = win.min();
            if (std::isnan(recent_high) || std::isnan(recent_low)) continue;
            double stop = recent_low - sl_buffer;
            double tp = b.close + tp_mult * (b.close - stop);
            if (b.close > recent_high) {
                out << std::chrono::duration_cast<std::chrono::seconds>(b.ts.time_since_epoch()).count() << ",UP," << stop << "," << tp << ",0.6\n";
                continue;
            }
            stop = recent_high + sl_buffer;
            tp = b.close - tp_mult * (stop - b.close);
            if (b.close < recent_low) {
                out << std::chrono::duration_cast<std::chrono::seconds>(b.ts.time_since_epoch()).count() << ",DOWN," << stop << "," << tp << ",0.6\n";
            }
        }
        std::cout << "Signals written: " << out_path << "\n";
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
