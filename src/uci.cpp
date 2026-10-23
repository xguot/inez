// SPDX-License-Identifier: GPL-3.0-only
#include "inez/uci.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace inez {
namespace {

std::string_view trim(std::string_view text)
{
	const auto begin = text.find_first_not_of(" \t\r\n");
	if (begin == std::string_view::npos)
		return {};
	const auto end = text.find_last_not_of(" \t\r\n");
	return text.substr(begin, end - begin + 1);
}

std::vector<std::string_view> split_tokens(std::string_view text)
{
	std::vector<std::string_view> tokens;
	std::size_t i = 0;
	while (i < text.size()) {
		while (i < text.size() && (text[i] == ' ' || text[i] == '\t' ||
		                           text[i] == '\r' || text[i] == '\n'))
			++i;
		if (i >= text.size())
			break;
		const std::size_t start = i;
		while (i < text.size() && text[i] != ' ' && text[i] != '\t' &&
		       text[i] != '\r' && text[i] != '\n')
			++i;
		tokens.push_back(text.substr(start, i - start));
	}
	return tokens;
}

template <class Int>
bool parse_int(std::string_view token, Int& out)
{
	if (token.empty())
		return false;
	const auto [ptr, ec] = std::from_chars(token.data(),
	                                       token.data() + token.size(), out);
	return ec == std::errc{} && ptr == token.data() + token.size();
}

bool parse_double(std::string_view token, double& out)
{
	try {
		std::string s(token);
		std::size_t pos = 0;
		const double v = std::stod(s, &pos);
		if (pos != s.size() || !std::isfinite(v))
			return false;
		out = v;
		return true;
	} catch (...) {
		return false;
	}
}

int win_rate_to_cp(double win_rate)
{
	const double clamped = std::clamp(2.0 * win_rate - 1.0, -0.999, 0.999);
	return static_cast<int>(std::lround(600.0 * std::atanh(clamped)));
}

std::string format_uci_info(const SearchInfo& info)
{
	std::ostringstream out;
	out << "info depth " << std::max(1, info.max_depth)
	    << " nodes " << info.nodes
	    << " nps " << static_cast<std::uint64_t>(std::lround(info.nodes_per_sec))
	    << " time " << info.elapsed.count()
	    << " score cp " << win_rate_to_cp(info.win_rate);
	if (!info.pv.empty()) {
		out << " pv";
		for (const Move& move : info.pv)
			out << ' ' << move.uci();
	}
	return out.str();
}

std::string format_diag_info(const SearchInfo& info)
{
	std::ostringstream out;
	out << std::fixed << std::setprecision(2)
	    << "diag nodes=" << info.nodes
	    << " depth=" << info.max_depth
	    << " nps=" << info.nodes_per_sec
	    << " batches=" << info.batches
	    << " leaves=" << info.batch_leaves
	    << " avg_batch=" << info.avg_batch_size
	    << " batch_util=" << info.batch_utilization
	    << " eval_us=" << info.avg_eval_latency_us;
	return out.str();
}

} // namespace

Engine::Engine(std::istream& in, std::ostream& out, std::ostream& log)
	: in_(in),
	  out_(out),
	  log_(log),
	  position_(Position::start()),
	  material_evaluator_(),
	  uniform_evaluator_(),
	  search_config_{},
	  search_(material_evaluator_, search_config_)
{
}

Engine::~Engine()
{
	wait_for_search();
}

void Engine::write(const std::string& line)
{
	std::lock_guard<std::mutex> lock(out_mutex_);
	out_ << line << '\n';
	out_.flush();
}

void Engine::log_line(const std::string& line)
{
	std::lock_guard<std::mutex> lock(log_mutex_);
	log_ << line << '\n';
	log_.flush();
}

void Engine::wait_for_search()
{
	search_.stop();
	if (search_thread_.joinable())
		search_thread_.join();
}

void Engine::handle_uci()
{
	write("id name Inezita 0.1.2");
	write("id author Inezita Developers");
	write("option name Evaluator type combo default material var material var uniform");
	write("option name BatchSize type spin default 1 min 1 max 256");
	write("option name Cpuct type string default 2.5");
	write("option name VirtualLoss type string default 1.0");
	write("option name InfoInterval type spin default 500 min 0 max 1000000");
	write("uciok");
}

void Engine::handle_isready()
{
	write("readyok");
}

void Engine::handle_ucinewgame()
{
	wait_for_search();
	position_ = Position::start();
}

void Engine::handle_setoption(std::string_view args)
{
	wait_for_search();
	const auto tokens = split_tokens(args);
	if (tokens.size() < 4 || tokens[0] != "name") {
		log_line("uci warning: malformed setoption command");
		return;
	}

	std::size_t value_idx = tokens.size();
	for (std::size_t i = 1; i < tokens.size(); ++i) {
		if (tokens[i] == "value") {
			value_idx = i;
			break;
		}
	}
	if (value_idx == 1 || value_idx + 1 >= tokens.size()) {
		log_line("uci warning: setoption requires name and value");
		return;
	}

	std::string name(tokens[1]);
	for (std::size_t i = 2; i < value_idx; ++i) {
		name += ' ';
		name += tokens[i];
	}
	std::string value(tokens[value_idx + 1]);
	for (std::size_t i = value_idx + 2; i < tokens.size(); ++i) {
		value += ' ';
		value += tokens[i];
	}

	if (name == "Evaluator") {
		if (value == "material") {
			search_.set_evaluator(material_evaluator_);
		} else if (value == "uniform") {
			search_.set_evaluator(uniform_evaluator_);
		} else {
			log_line("uci warning: unknown Evaluator value: " + value);
		}
	} else if (name == "BatchSize") {
		int batch_size = 1;
		if (parse_int(value, batch_size) && batch_size >= 1 && batch_size <= 256) {
			search_config_.batch_size = batch_size;
			search_.set_config(search_config_);
		} else {
			log_line("uci warning: invalid BatchSize: " + value);
		}
	} else if (name == "Cpuct") {
		double cpuct = 2.5;
		if (parse_double(value, cpuct) && cpuct >= 0.0) {
			search_config_.cpuct = cpuct;
			search_.set_config(search_config_);
		} else {
			log_line("uci warning: invalid Cpuct: " + value);
		}
	} else if (name == "VirtualLoss") {
		double vloss = 1.0;
		if (parse_double(value, vloss) && vloss >= 0.0) {
			search_config_.virtual_loss = vloss;
			search_.set_config(search_config_);
		} else {
			log_line("uci warning: invalid VirtualLoss: " + value);
		}
	} else if (name == "InfoInterval") {
		int interval = 500;
		if (parse_int(value, interval) && interval >= 0) {
			search_config_.info_interval = interval;
			search_.set_config(search_config_);
		} else {
			log_line("uci warning: invalid InfoInterval: " + value);
		}
	} else {
		log_line("uci warning: unknown option: " + name);
	}
}

void Engine::handle_position(std::string_view args)
{
	wait_for_search();
	const auto tokens = split_tokens(args);
	if (tokens.empty()) {
		log_line("uci error: position requires arguments");
		return;
	}

	try {
		Position next;
		std::size_t idx = 0;
		if (tokens[0] == "startpos") {
			next = Position::start();
			idx = 1;
		} else if (tokens[0] == "fen") {
			if (tokens.size() < 7)
				throw std::invalid_argument("position fen requires six FEN fields");
			std::string fen_str;
			for (std::size_t i = 1; i <= 6; ++i) {
				if (i > 1)
					fen_str += ' ';
				fen_str += tokens[i];
			}
			next = Position::from_fen(fen_str);
			idx = 7;
		} else {
			throw std::invalid_argument("position must start with startpos or fen");
		}

		if (idx < tokens.size()) {
			if (tokens[idx] != "moves")
				throw std::invalid_argument("expected 'moves' keyword in position command");
			++idx;
			for (; idx < tokens.size(); ++idx)
				next.play_uci(tokens[idx]);
		}
		position_ = std::move(next);
	} catch (const std::exception& error) {
		log_line(std::string("uci error in position: ") + error.what());
	}
}

void Engine::handle_go(std::string_view args)
{
	wait_for_search();

	SearchLimits limits;
	const auto tokens = split_tokens(args);
	for (std::size_t i = 0; i < tokens.size(); ++i) {
		const auto token = tokens[i];
		if (token == "infinite") {
			limits.infinite = true;
		} else if (token == "nodes" && i + 1 < tokens.size()) {
			int n = 0;
			if (parse_int(tokens[++i], n) && n >= 1)
				limits.nodes = n;
			else
				log_line("uci warning: invalid nodes limit");
		} else if (token == "movetime" && i + 1 < tokens.size()) {
			std::int64_t ms = 0;
			if (parse_int(tokens[++i], ms) && ms >= 1)
				limits.movetime = std::chrono::milliseconds(ms);
			else
				log_line("uci warning: invalid movetime limit");
		} else if (token == "wtime" && i + 1 < tokens.size()) {
			std::int64_t ms = 0;
			if (parse_int(tokens[++i], ms))
				limits.wtime = std::chrono::milliseconds(std::max<std::int64_t>(1, ms));
		} else if (token == "btime" && i + 1 < tokens.size()) {
			std::int64_t ms = 0;
			if (parse_int(tokens[++i], ms))
				limits.btime = std::chrono::milliseconds(std::max<std::int64_t>(1, ms));
		} else if (token == "winc" && i + 1 < tokens.size()) {
			std::int64_t ms = 0;
			if (parse_int(tokens[++i], ms))
				limits.winc = std::chrono::milliseconds(std::max<std::int64_t>(0, ms));
		} else if (token == "binc" && i + 1 < tokens.size()) {
			std::int64_t ms = 0;
			if (parse_int(tokens[++i], ms))
				limits.binc = std::chrono::milliseconds(std::max<std::int64_t>(0, ms));
		} else if (token == "movestogo" && i + 1 < tokens.size()) {
			int mtg = 0;
			if (parse_int(tokens[++i], mtg) && mtg >= 1)
				limits.movestogo = mtg;
		} else {
			log_line("uci warning: ignored go token: " + std::string(token));
		}
	}

	if (position_.outcome(false).is_terminal()) {
		log_line("uci warning: go called on terminal position");
		write("bestmove 0000");
		return;
	}

	const Position root = position_;
	search_thread_ = std::thread([this, root, limits]() {
		try {
			const SearchResult result = search_.run(
				root, limits, [this](const SearchInfo& info) {
					write(format_uci_info(info));
				});
			write(format_uci_info(result.info));
			log_line(format_diag_info(result.info));
			write("bestmove " + result.best_move.uci());
		} catch (const std::exception& error) {
			log_line(std::string("uci search error: ") + error.what());
			write("bestmove 0000");
		}
	});
}

void Engine::handle_stop()
{
	wait_for_search();
}

void Engine::run()
{
	std::string line;
	while (std::getline(in_, line)) {
		const std::string_view input = trim(line);
		if (input.empty())
			continue;
		const auto space = input.find_first_of(" \t");
		const std::string_view command = input.substr(0, space);
		const std::string_view args = space == std::string_view::npos
			? std::string_view{}
			: trim(input.substr(space + 1));

		if (command == "uci") {
			handle_uci();
		} else if (command == "isready") {
			handle_isready();
		} else if (command == "ucinewgame") {
			handle_ucinewgame();
		} else if (command == "setoption") {
			handle_setoption(args);
		} else if (command == "position") {
			handle_position(args);
		} else if (command == "go") {
			handle_go(args);
		} else if (command == "stop") {
			handle_stop();
		} else if (command == "quit") {
			break;
		} else {
			log_line("uci warning: unknown command: " + std::string(command));
		}
	}
	wait_for_search();
}

} // namespace inez
