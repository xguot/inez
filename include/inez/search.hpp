// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "inez/chess.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace inez {

// Position evaluation from the side to move's perspective.
struct Evaluation {
	// Expected outcome in [-1, 1].  +1 = side to move wins.
	double value{0.0};
	// Prior probability for each legal move.  policy[i] corresponds to
	// legal[i] as passed to evaluate().  Sum should be approximately 1.
	std::vector<double> policy;
};

struct BatchQuery {
	const Position* position{nullptr};
	const std::vector<Move>* legal{nullptr};
};

// Position evaluator interface.  The baseline implementation uses material
// and piece-square heuristics; a neural network replaces it in step 4.
class Evaluator {
public:
	virtual ~Evaluator() = default;
	// Evaluate the position and assign priors to the given legal moves.
	virtual Evaluation evaluate(const Position& position,
	                            const std::vector<Move>& legal) = 0;

	// Evaluate a batch of leaf positions.  The default implementation
	// calls evaluate() serially for each query; batched or stub evaluators
	// may override this to process multiple leaves in a single call.
	virtual std::vector<Evaluation> evaluate_batch(
		const std::vector<BatchQuery>& queries);
};

// Material count and piece-square tables, converted to a value in [-1, 1]
// via tanh.  Policy priors favour captures, promotions, and central moves.
class MaterialEvaluator : public Evaluator {
public:
	Evaluation evaluate(const Position& position,
	                    const std::vector<Move>& legal) override;
};

// Every legal move gets equal probability; value is always 0.
// Useful for testing search mechanics in isolation.
class UniformEvaluator : public Evaluator {
public:
	Evaluation evaluate(const Position& position,
	                    const std::vector<Move>& legal) override;
};

struct SearchLimits {
	std::optional<int> nodes;
	std::optional<std::chrono::milliseconds> movetime;
	std::optional<std::chrono::milliseconds> wtime;
	std::optional<std::chrono::milliseconds> btime;
	std::optional<std::chrono::milliseconds> winc;
	std::optional<std::chrono::milliseconds> binc;
	std::optional<int> movestogo;
	bool infinite{false};
};

struct SearchInfo {
	int nodes{0};
	int max_depth{0};
	// Win probability from the root side's perspective, in [0, 1].
	double win_rate{0.5};
	std::chrono::milliseconds elapsed{0};
	std::vector<Move> pv;
	// Throughput, batching, and leaf-evaluation latency diagnostics.
	double nodes_per_sec{0.0};
	int batches{0};
	int batch_leaves{0};
	double avg_batch_size{0.0};
	double batch_utilization{0.0};
	double avg_eval_latency_us{0.0};
};

// Periodic search progress callback.  May be called from the search
// thread; implementations must be safe for concurrent use.
using InfoCallback = std::function<void(const SearchInfo&)>;

struct SearchResult {
	Move best_move;
	SearchInfo info;
};

struct SearchConfig {
	double cpuct{2.5};
	// Report info every this many simulations.
	int info_interval{500};
	// Maximum number of leaves evaluated together per batch (1 = serial).
	int batch_size{1};
	// Virtual loss penalty applied per in-flight visit during batching.
	double virtual_loss{1.0};
};

class Search {
public:
	explicit Search(Evaluator& evaluator, SearchConfig config = {});
	~Search();

	Search(const Search&) = delete;
	Search& operator=(const Search&) = delete;

	void set_evaluator(Evaluator& evaluator) noexcept { evaluator_ = &evaluator; }
	void set_config(SearchConfig config) noexcept { config_ = config; }
	[[nodiscard]] const SearchConfig& config() const noexcept { return config_; }

	// Run MCTS from the given root position until limits are reached or
	// stop() is called.  Returns the best move and search statistics.
	// The position is not modified.  Throws std::logic_error if the root
	// is terminal.
	SearchResult run(const Position& root, const SearchLimits& limits,
	                 InfoCallback on_info = {});

	// Signal the running search to finish after the current simulation.
	// Thread-safe; may be called from any thread.
	void stop() noexcept;

	// Compute the time budget for a single move under the given limits.
	[[nodiscard]] std::chrono::milliseconds allocate_time(
		const SearchLimits& limits, Color side) const;

private:
	struct Node;
	struct Edge {
		Move move;
		double prior{0.0};
		int visit_count{0};
		int in_flight{0};
		// Accumulated value from the parent's side-to-move perspective.
		double value_sum{0.0};
		Node* child{nullptr};
	};
	struct Node {
		std::vector<Edge> edges;
		bool expanded{false};
		bool pending{false};
		bool terminal{false};
		double initial_value{0.0};
		// Terminal value from this node's side-to-move perspective.
		double terminal_value{0.0};
	};

	Evaluator* evaluator_;
	SearchConfig config_;
	std::atomic<bool> stop_{false};
	std::vector<std::unique_ptr<Node>> arena_;
	int nodes_{0};
	int max_depth_{0};
	int batches_{0};
	int batch_leaves_{0};
	std::uint64_t eval_time_ns_{0};
	double root_value_{0.0};

	Node* new_node();
	void clear_tree();
	Edge& select_child(Node& node) const;
	void expand_node(Node& node, const std::vector<Move>& legal,
	                 const std::vector<double>& policy);
	SearchInfo gather_info(const Node& root,
	                       std::chrono::steady_clock::time_point start) const;
};

} // namespace inez
