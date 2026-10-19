// SPDX-License-Identifier: GPL-3.0-only
#include "inez/chess.hpp"
#include "inez/search.hpp"
#include "inez/uci.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using inez::BatchQuery;
using inez::Color;
using inez::Engine;
using inez::Evaluation;
using inez::Evaluator;
using inez::MaterialEvaluator;
using inez::Move;
using inez::Position;
using inez::Search;
using inez::SearchConfig;
using inez::SearchInfo;
using inez::SearchLimits;
using inez::UniformEvaluator;

void require(bool condition, std::string_view expression, int line)
{
	if (!condition)
		throw std::runtime_error("line " + std::to_string(line) + ": " +
		                         std::string(expression));
}

#define REQUIRE(expression) require((expression), #expression, __LINE__)

template <class Exception, class Function>
void throws(Function&& function)
{
	bool caught = false;
	try {
		function();
	} catch (const Exception&) {
		caught = true;
	}
	REQUIRE(caught);
}

class StubBatchEvaluator final : public Evaluator {
public:
	Evaluation evaluate(const Position& position,
	                    const std::vector<Move>& legal) override
	{
		return fallback_.evaluate(position, legal);
	}

	std::vector<Evaluation> evaluate_batch(
		const std::vector<BatchQuery>& queries) override
	{
		batch_sizes.push_back(static_cast<int>(queries.size()));
		for (std::size_t i = 0; i < queries.size(); ++i) {
			REQUIRE(queries[i].position != nullptr);
			REQUIRE(queries[i].legal != nullptr);
			for (std::size_t j = i + 1; j < queries.size(); ++j) {
				// Within a single batch, virtual loss must route trajectories
				// to distinct unexpanded leaf positions.
				REQUIRE(!(queries[i].position->search_key() ==
				          queries[j].position->search_key()));
			}
		}
		return Evaluator::evaluate_batch(queries);
	}

	std::vector<int> batch_sizes;

private:
	MaterialEvaluator fallback_;
};

void evaluators_and_priors()
{
	UniformEvaluator uniform;
	const auto start = Position::start();
	const auto legal = start.legal_moves();
	const auto u_eval = uniform.evaluate(start, legal);
	REQUIRE(u_eval.value == 0.0);
	REQUIRE(u_eval.policy.size() == legal.size());
	double sum = 0.0;
	for (double p : u_eval.policy) {
		REQUIRE(std::abs(p - 1.0 / static_cast<double>(legal.size())) < 1e-12);
		sum += p;
	}
	REQUIRE(std::abs(sum - 1.0) < 1e-12);
	REQUIRE(uniform.evaluate(start, {}).policy.empty());

	MaterialEvaluator material;
	const auto start_eval = material.evaluate(start, legal);
	REQUIRE(std::abs(start_eval.value) < 1e-12);
	REQUIRE(start_eval.policy.size() == legal.size());
	double mat_sum = 0.0;
	for (double p : start_eval.policy) {
		REQUIRE(p > 0.0);
		mat_sum += p;
	}
	REQUIRE(std::abs(mat_sum - 1.0) < 1e-9);

	const auto white_up = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
	const auto black_down = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 b - - 0 1");
	const double w_val = material.evaluate(white_up, white_up.legal_moves()).value;
	const double b_val = material.evaluate(black_down, black_down.legal_moves()).value;
	REQUIRE(w_val > 0.8);
	REQUIRE(b_val < -0.8);
	REQUIRE(std::abs(w_val + b_val) < 1e-12);

	// Capture and promotion priors should exceed quiet fallback moves.
	const auto cap_pos = Position::from_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1");
	const auto cap_moves = cap_pos.legal_moves();
	const auto cap_eval = material.evaluate(cap_pos, cap_moves);
	auto find_prior = [&](std::string_view uci) {
		for (std::size_t i = 0; i < cap_moves.size(); ++i) {
			if (cap_moves[i].uci() == uci)
				return cap_eval.policy[i];
		}
		throw std::runtime_error("missing move " + std::string(uci));
	};
	REQUIRE(find_prior("e4d5") > find_prior("e4e5"));
}

void serial_puct_traversal_and_terminal_scoring()
{
	UniformEvaluator uniform;
	Search search(uniform, SearchConfig{.cpuct = 2.0, .info_interval = 10, .batch_size = 1});

	const auto mated = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
	const auto saved_mated = mated;
	throws<std::logic_error>([&] {
		(void)search.run(mated, SearchLimits{.nodes = 10});
	});
	REQUIRE(mated == saved_mated);

	// Mate in 1 with UniformEvaluator: once all 10 legal moves are tried,
	// e1e8# scores +1.0 while non-mating moves score 0.0.
	const auto mate_in_1 = Position::from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");
	const auto saved_mate = mate_in_1;
	int callbacks = 0;
	const auto result = search.run(
		mate_in_1, SearchLimits{.nodes = 60},
		[&](const SearchInfo& info) {
			++callbacks;
			REQUIRE(info.nodes >= 10);
			REQUIRE(!info.pv.empty());
		});
	REQUIRE(mate_in_1 == saved_mate);
	REQUIRE(callbacks >= 5);
	REQUIRE(result.best_move.uci() == "e1e8");
	REQUIRE(result.info.nodes == 60);
	REQUIRE(result.info.win_rate > 0.95);
	REQUIRE(!result.info.pv.empty());
	REQUIRE(result.info.pv.front().uci() == "e1e8");

	// Stalemate vs checkmate: g5g6 is stalemate (0.0), g5g7# is mate (+1.0).
	MaterialEvaluator material;
	Search mat_search(material);
	const auto queen_mate = Position::from_fen("7k/5K2/8/6Q1/8/8/8/8 w - - 0 1");
	const auto q_res = mat_search.run(queen_mate, SearchLimits{.nodes = 50});
	REQUIRE(q_res.best_move.uci() == "g5g7");
	REQUIRE(q_res.info.win_rate > 0.95);

	// Repetition avoidance when winning: play moves so repeating once more
	// triggers a 3-fold draw inside the search tree.
	auto rep_pos = Position::from_fen("6k1/5ppp/8/8/8/8/1Q6/6K1 w - - 0 1");
	for (const auto uci : {"b2b1", "g8h8", "b1b2", "h8g8", "b2b1", "g8h8"})
		rep_pos.play_uci(uci);
	// Now b1b2 repeats the initial position for the 3rd time (draw = 0.0),
	// whereas White is up a full queen (+0.9) and has b1b8# or other winning moves.
	const auto rep_res = mat_search.run(rep_pos, SearchLimits{.nodes = 80});
	REQUIRE(rep_res.best_move.uci() != "b1b2");
}

void virtual_loss_and_batching()
{
	StubBatchEvaluator stub;
	Search search(stub, SearchConfig{
		.cpuct = 2.5,
		.info_interval = 16,
		.batch_size = 8,
		.virtual_loss = 1.0,
	});

	const auto start = Position::start();
	const auto result = search.run(start, SearchLimits{.nodes = 33});
	REQUIRE(result.info.nodes == 33);
	REQUIRE(result.info.batches == 5);
	REQUIRE(result.info.batch_leaves == 33);
	REQUIRE(stub.batch_sizes.size() == 5);
	REQUIRE(stub.batch_sizes[0] == 1);
	for (std::size_t i = 1; i < stub.batch_sizes.size(); ++i)
		REQUIRE(stub.batch_sizes[i] == 8);
	REQUIRE(std::abs(result.info.avg_batch_size - 33.0 / 5.0) < 1e-9);
	REQUIRE(result.info.batch_utilization > 0.8);
	REQUIRE(result.info.avg_eval_latency_us >= 0.0);
	REQUIRE(result.info.nodes_per_sec > 0.0);
	REQUIRE(!result.info.pv.empty());

	// Also verify batched search on a position with immediate terminal children.
	stub.batch_sizes.clear();
	const auto mate_in_1 = Position::from_fen("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1");
	const auto mate_res = search.run(mate_in_1, SearchLimits{.nodes = 40});
	REQUIRE(mate_res.best_move.uci() == "e1e8");
	REQUIRE(mate_res.info.nodes == 40);
}

void clock_management_and_stop()
{
	using namespace std::chrono_literals;
	MaterialEvaluator material;
	Search search(material);

	REQUIRE(search.allocate_time(SearchLimits{.movetime = 250ms}, Color::White) == 250ms);
	const auto blitz = search.allocate_time(
		SearchLimits{.wtime = 60000ms, .btime = 60000ms, .winc = 1000ms, .binc = 1000ms},
		Color::White);
	REQUIRE(blitz >= 1000ms && blitz <= 5000ms);

	const auto sudden_death_last = search.allocate_time(
		SearchLimits{.wtime = 500ms, .movestogo = 1}, Color::White);
	REQUIRE(sudden_death_last == 480ms);

	const auto tiny = search.allocate_time(
		SearchLimits{.btime = 5ms}, Color::Black);
	REQUIRE(tiny == 1ms);

	REQUIRE(search.allocate_time(SearchLimits{.infinite = true}, Color::White) ==
	        std::chrono::milliseconds::max());

	// Asynchronous stop() from another thread during infinite search.
	const auto start = Position::start();
	std::thread stopper([&] {
		std::this_thread::sleep_for(15ms);
		search.stop();
	});
	const auto res = search.run(start, SearchLimits{.infinite = true});
	stopper.join();
	REQUIRE(res.info.nodes >= 1);
	REQUIRE(!res.info.pv.empty());
}

void tactical_and_endgame_suite()
{
	MaterialEvaluator material;
	Search search(material, SearchConfig{.cpuct = 2.0, .batch_size = 4});

	struct Tactic {
		std::string_view name;
		std::string_view fen;
		std::string_view expected_uci;
		int nodes;
	};
	const std::vector<Tactic> suite = {
		{"back-rank mate in 1", "6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1", "d1d8", 80},
		{"queen mate in 1", "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1", "g5g7", 80},
		{"capture undefended queen", "4k3/8/8/3q4/4N3/8/8/4K3 w - - 0 1", "e4d6", -1},
		{"free queen capture", "4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1", "c3d5", 100},
		{"promote to queen with mate", "7k/4P1pp/8/8/8/8/8/6K1 w - - 0 1", "e7e8q", 100},
	};

	for (const auto& item : suite) {
		if (item.nodes < 0)
			continue;
		const auto pos = Position::from_fen(item.fen);
		const auto res = search.run(pos, SearchLimits{.nodes = item.nodes});
		if (res.best_move.uci() != item.expected_uci) {
			throw std::runtime_error(std::string(item.name) + ": expected " +
			                         std::string(item.expected_uci) + ", got " +
			                         res.best_move.uci());
		}
	}
}

void uci_protocol_and_stream_separation()
{
	std::istringstream input(
		"uci\n"
		"isready\n"
		"setoption name Evaluator value uniform\n"
		"setoption name Evaluator value material\n"
		"setoption name BatchSize value 4\n"
		"setoption name Cpuct value 2.2\n"
		"setoption name VirtualLoss value 1.0\n"
		"setoption name InfoInterval value 20\n"
		"ucinewgame\n"
		"position startpos moves e2e4 e7e5\n"
		"go nodes 40\n"
		"isready\n"
		"position fen 6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1\n"
		"go nodes 60\n"
		"isready\n"
		"unknowncommand\n"
		"position invalid_fen\n"
		"quit\n");
	std::ostringstream output;
	std::ostringstream diagnostics;

	{
		Engine engine(input, output, diagnostics);
		engine.run();
	}

	const std::string out_text = output.str();
	const std::string log_text = diagnostics.str();

	REQUIRE(out_text.find("id name Inez") != std::string::npos);
	REQUIRE(out_text.find("uciok\n") != std::string::npos);
	REQUIRE(out_text.find("readyok\n") != std::string::npos);
	REQUIRE(out_text.find("bestmove e1e8\n") != std::string::npos);
	// Protocol output must never contain diagnostic or warning lines.
	REQUIRE(out_text.find("diag ") == std::string::npos);
	REQUIRE(out_text.find("uci warning") == std::string::npos);
	REQUIRE(out_text.find("uci error") == std::string::npos);

	// Diagnostics stream receives batch telemetry and command warnings.
	REQUIRE(log_text.find("diag nodes=") != std::string::npos);
	REQUIRE(log_text.find("uci warning: unknown command: unknowncommand") !=
	        std::string::npos);
	REQUIRE(log_text.find("uci error in position:") != std::string::npos);
}

} // namespace

int main()
{
	const std::vector<std::pair<std::string_view, std::function<void()>>> tests = {
		{"evaluators and policy priors", evaluators_and_priors},
		{"serial PUCT traversal and terminal scoring", serial_puct_traversal_and_terminal_scoring},
		{"virtual loss and leaf batching with stub", virtual_loss_and_batching},
		{"clock management and asynchronous stop", clock_management_and_stop},
		{"tactical and endgame suite", tactical_and_endgame_suite},
		{"UCI protocol and diagnostic separation", uci_protocol_and_stream_separation},
	};
	int failed = 0;
	for (const auto& [name, test] : tests) {
		try {
			test();
			std::cout << "PASS " << name << '\n';
		} catch (const std::exception& error) {
			++failed;
			std::cerr << "FAIL " << name << ": " << error.what() << '\n';
		}
	}
	if (failed != 0)
		std::cerr << failed << " test group(s) failed\n";
	return failed == 0 ? 0 : 1;
}
