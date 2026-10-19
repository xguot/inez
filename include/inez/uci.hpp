// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "inez/chess.hpp"
#include "inez/search.hpp"

#include <iosfwd>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

namespace inez {

// UCI protocol handler.  Reads commands from `in`, writes protocol
// output to `out`, and diagnostic messages to `log`.  Search runs in
// a background thread so that `stop` can interrupt it.
class Engine {
public:
	Engine(std::istream& in, std::ostream& out, std::ostream& log);
	~Engine();

	Engine(const Engine&) = delete;
	Engine& operator=(const Engine&) = delete;

	// Block on stdin until `quit` is received or the stream ends.
	void run();

private:
	void handle_uci();
	void handle_isready();
	void handle_ucinewgame();
	void handle_setoption(std::string_view args);
	void handle_position(std::string_view args);
	void handle_go(std::string_view args);
	void handle_stop();
	void wait_for_search();
	void write(const std::string& line);
	void log_line(const std::string& line);

	std::istream& in_;
	std::ostream& out_;
	std::ostream& log_;
	std::mutex out_mutex_;
	std::mutex log_mutex_;
	Position position_;
	MaterialEvaluator material_evaluator_;
	UniformEvaluator uniform_evaluator_;
	SearchConfig search_config_{};
	Search search_;
	std::thread search_thread_;
};

} // namespace inez
