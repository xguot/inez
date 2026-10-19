// SPDX-License-Identifier: GPL-3.0-only
#include "inez/search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace inez {
namespace {

constexpr int pawn_value = 100;
constexpr int knight_value = 320;
constexpr int bishop_value = 330;
constexpr int rook_value = 500;
constexpr int queen_value = 900;
constexpr int bishop_pair_bonus = 30;

// Piece-square tables indexed from White's perspective (a1 = 0 .. h8 = 63).
constexpr std::array<int, 64> pawn_pst{{
	  0,   0,   0,   0,   0,   0,   0,   0,
	  5,  10,  10, -20, -20,  10,  10,   5,
	  5,  -5, -10,   0,   0, -10,  -5,   5,
	  0,   0,   0,  20,  20,   0,   0,   0,
	  5,   5,  10,  25,  25,  10,   5,   5,
	 10,  10,  20,  30,  30,  20,  10,  10,
	 50,  50,  50,  50,  50,  50,  50,  50,
	  0,   0,   0,   0,   0,   0,   0,   0,
}};

constexpr std::array<int, 64> knight_pst{{
	-50, -40, -30, -30, -30, -30, -40, -50,
	-40, -20,   0,   5,   5,   0, -20, -40,
	-30,   5,  10,  15,  15,  10,   5, -30,
	-30,   0,  15,  20,  20,  15,   0, -30,
	-30,   5,  15,  20,  20,  15,   5, -30,
	-30,   0,  10,  15,  15,  10,   0, -30,
	-40, -20,   0,   0,   0,   0, -20, -40,
	-50, -40, -30, -30, -30, -30, -40, -50,
}};

constexpr std::array<int, 64> bishop_pst{{
	-20, -10, -10, -10, -10, -10, -10, -20,
	-10,   5,   0,   0,   0,   0,   5, -10,
	-10,  10,  10,  10,  10,  10,  10, -10,
	-10,   0,  10,  10,  10,  10,   0, -10,
	-10,   5,   5,  10,  10,   5,   5, -10,
	-10,   0,   5,  10,  10,   5,   0, -10,
	-10,   0,   0,   0,   0,   0,   0, -10,
	-20, -10, -10, -10, -10, -10, -10, -20,
}};

constexpr std::array<int, 64> rook_pst{{
	  0,   0,   0,   5,   5,   0,   0,   0,
	 -5,   0,   0,   0,   0,   0,   0,  -5,
	 -5,   0,   0,   0,   0,   0,   0,  -5,
	 -5,   0,   0,   0,   0,   0,   0,  -5,
	 -5,   0,   0,   0,   0,   0,   0,  -5,
	 -5,   0,   0,   0,   0,   0,   0,  -5,
	  5,  10,  10,  10,  10,  10,  10,   5,
	  0,   0,   0,   0,   0,   0,   0,   0,
}};

constexpr std::array<int, 64> queen_pst{{
	-20, -10, -10,  -5,  -5, -10, -10, -20,
	-10,   0,   5,   0,   0,   0,   0, -10,
	-10,   5,   5,   5,   5,   5,   0, -10,
	  0,   0,   5,   5,   5,   5,   0,  -5,
	 -5,   0,   5,   5,   5,   5,   0,  -5,
	-10,   0,   5,   5,   5,   5,   0, -10,
	-10,   0,   0,   0,   0,   0,   0, -10,
	-20, -10, -10,  -5,  -5, -10, -10, -20,
}};

constexpr std::array<int, 64> king_pst{{
	 20,  30,  10,   0,   0,  10,  30,  20,
	 20,  20,   0,   0,   0,   0,  20,  20,
	-10, -20, -20, -20, -20, -20, -20, -10,
	-20, -30, -30, -40, -40, -30, -30, -20,
	-30, -40, -40, -50, -50, -40, -40, -30,
	-30, -40, -40, -50, -50, -40, -40, -30,
	-30, -40, -40, -50, -50, -40, -40, -30,
	-30, -40, -40, -50, -50, -40, -40, -30,
}};

int piece_value(PieceType type)
{
	switch (type) {
	case PieceType::Pawn: return pawn_value;
	case PieceType::Knight: return knight_value;
	case PieceType::Bishop: return bishop_value;
	case PieceType::Rook: return rook_value;
	case PieceType::Queen: return queen_value;
	case PieceType::King:
	case PieceType::None:
		return 0;
	}
	return 0;
}

int pst_bonus(PieceType type, int square, Color color)
{
	const int oriented = color == Color::White ? square : (square ^ 56);
	switch (type) {
	case PieceType::Pawn: return pawn_pst[oriented];
	case PieceType::Knight: return knight_pst[oriented];
	case PieceType::Bishop: return bishop_pst[oriented];
	case PieceType::Rook: return rook_pst[oriented];
	case PieceType::Queen: return queen_pst[oriented];
	case PieceType::King: return king_pst[oriented];
	case PieceType::None: return 0;
	}
	return 0;
}

int chebyshev_distance(int sq_a, int sq_b)
{
	const int df = std::abs((sq_a % 8) - (sq_b % 8));
	const int dr = std::abs((sq_a / 8) - (sq_b / 8));
	return std::max(df, dr);
}

std::vector<double> normalize_priors(const std::vector<double>& raw,
                                     std::size_t count)
{
	if (count == 0)
		return {};
	if (raw.size() != count)
		return std::vector<double>(count, 1.0 / static_cast<double>(count));

	std::vector<double> priors(count, 0.0);
	double sum = 0.0;
	for (std::size_t i = 0; i < count; ++i) {
		const double p = std::isfinite(raw[i]) && raw[i] > 0.0 ? raw[i] : 0.0;
		priors[i] = p;
		sum += p;
	}
	if (sum <= 0.0)
		return std::vector<double>(count, 1.0 / static_cast<double>(count));
	for (double& p : priors)
		p /= sum;
	return priors;
}

} // namespace

std::vector<Evaluation> Evaluator::evaluate_batch(
	const std::vector<BatchQuery>& queries)
{
	std::vector<Evaluation> results;
	results.reserve(queries.size());
	for (const BatchQuery& query : queries) {
		if (query.position == nullptr || query.legal == nullptr)
			throw std::invalid_argument("null BatchQuery pointer");
		results.push_back(evaluate(*query.position, *query.legal));
	}
	return results;
}

Evaluation UniformEvaluator::evaluate(const Position& /*position*/,
                                      const std::vector<Move>& legal)
{
	Evaluation eval;
	eval.value = 0.0;
	if (!legal.empty()) {
		const double uniform = 1.0 / static_cast<double>(legal.size());
		eval.policy.assign(legal.size(), uniform);
	}
	return eval;
}

Evaluation MaterialEvaluator::evaluate(const Position& position,
                                       const std::vector<Move>& legal)
{
	int white_score = 0;
	int black_score = 0;
	int white_bishops = 0;
	int black_bishops = 0;
	int white_king_sq = 4;
	int black_king_sq = 60;

	for (int sq = 0; sq < 64; ++sq) {
		const Piece piece = position.piece_at(sq);
		if (piece.type == PieceType::None)
			continue;
		const int score = piece_value(piece.type) +
			pst_bonus(piece.type, sq, piece.color);
		if (piece.color == Color::White) {
			white_score += score;
			if (piece.type == PieceType::Bishop)
				++white_bishops;
			else if (piece.type == PieceType::King)
				white_king_sq = sq;
		} else {
			black_score += score;
			if (piece.type == PieceType::Bishop)
				++black_bishops;
			else if (piece.type == PieceType::King)
				black_king_sq = sq;
		}
	}
	if (white_bishops >= 2)
		white_score += bishop_pair_bonus;
	if (black_bishops >= 2)
		black_score += bishop_pair_bonus;

	const Color mover = position.side_to_move();
	const int enemy_king_sq = mover == Color::White ? black_king_sq : white_king_sq;
	const int perspective_cp = mover == Color::White
		? (white_score - black_score)
		: (black_score - white_score);

	Evaluation eval;
	eval.value = std::tanh(static_cast<double>(perspective_cp) / 600.0);

	if (legal.empty())
		return eval;

	std::vector<double> logits(legal.size(), 0.0);
	double max_logit = -std::numeric_limits<double>::infinity();
	for (std::size_t i = 0; i < legal.size(); ++i) {
		const Move move = legal[i];
		const Piece moving = position.piece_at(move.from);
		const Piece target = position.piece_at(move.to);
		double logit = 0.0;

		const bool is_ep = moving.type == PieceType::Pawn &&
			target.type == PieceType::None &&
			(move.from % 8 != move.to % 8);
		if (target.type != PieceType::None || is_ep) {
			const int captured = is_ep ? pawn_value : piece_value(target.type);
			const int attacker = piece_value(moving.type);
			logit += 1.2 + 0.0025 * static_cast<double>(captured) -
				0.0004 * static_cast<double>(attacker);
		}

		if (move.promotion != PieceType::None) {
			switch (move.promotion) {
			case PieceType::Queen: logit += 2.0; break;
			case PieceType::Knight: logit += 0.9; break;
			case PieceType::Rook: logit += 0.5; break;
			case PieceType::Bishop: logit += 0.4; break;
			default: break;
			}
		}

		const int delta_pst = pst_bonus(moving.type, move.to, mover) -
			pst_bonus(moving.type, move.from, mover);
		logit += 0.015 * static_cast<double>(delta_pst);

		if (moving.type != PieceType::Pawn && moving.type != PieceType::King) {
			const int dist_before = chebyshev_distance(move.from, enemy_king_sq);
			const int dist_after = chebyshev_distance(move.to, enemy_king_sq);
			logit += 0.05 * static_cast<double>(dist_before - dist_after);
		}

		logits[i] = logit;
		max_logit = std::max(max_logit, logit);
	}

	eval.policy.resize(legal.size());
	double sum = 0.0;
	for (std::size_t i = 0; i < legal.size(); ++i) {
		eval.policy[i] = std::exp(logits[i] - max_logit);
		sum += eval.policy[i];
	}
	for (double& p : eval.policy)
		p /= sum;
	return eval;
}

Search::Search(Evaluator& evaluator, SearchConfig config)
	: evaluator_(&evaluator), config_(config)
{
}

Search::~Search() = default;

Search::Node* Search::new_node()
{
	arena_.push_back(std::make_unique<Node>());
	return arena_.back().get();
}

void Search::clear_tree()
{
	arena_.clear();
	nodes_ = 0;
	max_depth_ = 0;
	batches_ = 0;
	batch_leaves_ = 0;
	eval_time_ns_ = 0;
	root_value_ = 0.0;
}

void Search::stop() noexcept
{
	stop_.store(true, std::memory_order_relaxed);
}

std::chrono::milliseconds Search::allocate_time(const SearchLimits& limits,
                                                Color side) const
{
	using namespace std::chrono_literals;
	if (limits.movetime.has_value())
		return std::max(1ms, *limits.movetime);

	const auto remaining_opt = side == Color::White ? limits.wtime : limits.btime;
	const auto inc_opt = side == Color::White ? limits.winc : limits.binc;
	if (remaining_opt.has_value()) {
		constexpr auto overhead = 20ms;
		const auto remaining = *remaining_opt;
		const auto available = remaining > overhead ? (remaining - overhead) : 1ms;
		const int mtg = (limits.movestogo.has_value() && *limits.movestogo > 0)
			? *limits.movestogo
			: 30;
		const auto inc = std::max(0ms, inc_opt.value_or(0ms));
		const auto target = available / mtg + (inc * 3) / 4;
		const auto cap = mtg == 1 ? available : std::max(1ms, available / 2);
		return std::clamp(target, 1ms, cap);
	}

	if (!limits.nodes.has_value() && !limits.infinite)
		return 100ms;

	return std::chrono::milliseconds::max();
}

Search::Edge& Search::select_child(Node& node) const
{
	if (node.edges.empty())
		throw std::logic_error("cannot select child from unexpanded node");

	int total_visits = 0;
	for (const Edge& edge : node.edges)
		total_visits += edge.visit_count + edge.in_flight;

	const double sqrt_total = std::sqrt(static_cast<double>(std::max(1, total_visits)));
	const double cpuct = config_.cpuct;
	const double vloss = config_.virtual_loss;
	const double fpu = total_visits > 0
		? std::clamp(node.initial_value - 0.15, -1.0, 1.0)
		: 0.0;

	Edge* best = &node.edges.front();
	double best_score = -std::numeric_limits<double>::infinity();

	for (Edge& edge : node.edges) {
		const int eff_visits = edge.visit_count + edge.in_flight;
		const double q = eff_visits > 0
			? (edge.value_sum - vloss * static_cast<double>(edge.in_flight)) /
				static_cast<double>(eff_visits)
			: fpu;
		const double u = cpuct * edge.prior * sqrt_total /
			static_cast<double>(1 + eff_visits);
		const double score = q + u;
		if (score > best_score) {
			best_score = score;
			best = &edge;
		}
	}
	return *best;
}

void Search::expand_node(Node& node, const std::vector<Move>& legal,
                         const std::vector<double>& policy)
{
	const auto priors = normalize_priors(policy, legal.size());
	node.edges.resize(legal.size());
	for (std::size_t i = 0; i < legal.size(); ++i) {
		node.edges[i].move = legal[i];
		node.edges[i].prior = priors[i];
	}
	node.pending = false;
	node.expanded = true;
}

SearchInfo Search::gather_info(
	const Node& root, std::chrono::steady_clock::time_point start) const
{
	const auto now = std::chrono::steady_clock::now();
	const auto elapsed_ns = std::max<std::int64_t>(
		1, std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
	const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		now - start);

	SearchInfo info;
	info.nodes = nodes_;
	info.max_depth = max_depth_;
	info.elapsed = elapsed_ms;
	info.nodes_per_sec = static_cast<double>(nodes_) * 1e9 /
		static_cast<double>(elapsed_ns);
	info.batches = batches_;
	info.batch_leaves = batch_leaves_;
	const int configured_batch = std::max(1, config_.batch_size);
	if (batches_ > 0) {
		info.avg_batch_size = static_cast<double>(batch_leaves_) /
			static_cast<double>(batches_);
		info.batch_utilization = std::clamp(
			info.avg_batch_size / static_cast<double>(configured_batch),
			0.0, 1.0);
		info.avg_eval_latency_us = (static_cast<double>(eval_time_ns_) / 1000.0) /
			static_cast<double>(batches_);
	}

	const Edge* best_root_edge = nullptr;
	for (const Edge& edge : root.edges) {
		if (best_root_edge == nullptr) {
			best_root_edge = &edge;
			continue;
		}
		if (edge.visit_count != best_root_edge->visit_count) {
			if (edge.visit_count > best_root_edge->visit_count)
				best_root_edge = &edge;
			continue;
		}
		const double q_cur = edge.visit_count > 0
			? edge.value_sum / static_cast<double>(edge.visit_count)
			: 0.0;
		const double q_best = best_root_edge->visit_count > 0
			? best_root_edge->value_sum / static_cast<double>(best_root_edge->visit_count)
			: 0.0;
		if (q_cur > q_best || (q_cur == q_best && edge.prior > best_root_edge->prior))
			best_root_edge = &edge;
	}

	if (best_root_edge != nullptr && best_root_edge->visit_count > 0) {
		const double q = best_root_edge->value_sum /
			static_cast<double>(best_root_edge->visit_count);
		info.win_rate = std::clamp(0.5 * (q + 1.0), 0.0, 1.0);
	} else {
		info.win_rate = std::clamp(0.5 * (root_value_ + 1.0), 0.0, 1.0);
	}

	const Node* current = &root;
	while (current != nullptr && !current->edges.empty()) {
		const Edge* best = nullptr;
		for (const Edge& edge : current->edges) {
			if (edge.visit_count <= 0)
				continue;
			if (best == nullptr || edge.visit_count > best->visit_count) {
				best = &edge;
			} else if (edge.visit_count == best->visit_count) {
				const double q_edge = edge.value_sum / static_cast<double>(edge.visit_count);
				const double q_best = best->value_sum / static_cast<double>(best->visit_count);
				if (q_edge > q_best || (q_edge == q_best && edge.prior > best->prior))
					best = &edge;
			}
		}
		if (best == nullptr)
			break;
		info.pv.push_back(best->move);
		current = best->child;
	}
	return info;
}

SearchResult Search::run(const Position& root, const SearchLimits& limits,
                         InfoCallback on_info)
{
	if (root.outcome(false).is_terminal())
		throw std::logic_error("root position is terminal");

	stop_.store(false, std::memory_order_relaxed);
	clear_tree();

	const auto start_time = std::chrono::steady_clock::now();
	const auto budget = allocate_time(limits, root.side_to_move());
	const bool has_deadline = budget != std::chrono::milliseconds::max();
	const auto deadline = has_deadline ? (start_time + budget)
	                                   : std::chrono::steady_clock::time_point::max();

	Node* root_node = new_node();
	const int configured_batch = std::max(1, config_.batch_size);
	int last_reported_bucket = 0;

	struct Trajectory {
		std::vector<Edge*> path;
		Node* leaf{nullptr};
		Position position;
		std::vector<Move> legal;
		int depth{0};
		double value{0.0};
		bool needs_eval{false};
	};

	while (!stop_.load(std::memory_order_relaxed)) {
		if (limits.nodes.has_value() && nodes_ >= *limits.nodes)
			break;
		if (nodes_ > 0 && has_deadline &&
		    std::chrono::steady_clock::now() >= deadline)
			break;

		int target_batch = configured_batch;
		if (limits.nodes.has_value()) {
			const int remaining = *limits.nodes - nodes_;
			if (remaining <= 0)
				break;
			target_batch = std::min(target_batch, remaining);
		}

		std::vector<Trajectory> trajectories;
		trajectories.reserve(target_batch);

		for (int b = 0; b < target_batch; ++b) {
			if (b > 0 && stop_.load(std::memory_order_relaxed))
				break;
			if (b > 0 && has_deadline &&
			    std::chrono::steady_clock::now() >= deadline)
				break;

			Trajectory traj{
				.path = {},
				.leaf = root_node,
				.position = root,
				.legal = {},
				.depth = 0,
				.value = 0.0,
				.needs_eval = false,
			};

			while (traj.leaf->expanded && !traj.leaf->terminal) {
				Edge& edge = select_child(*traj.leaf);
				++edge.in_flight;
				traj.path.push_back(&edge);
				traj.position.make_move(edge.move);
				if (edge.child == nullptr)
					edge.child = new_node();
				traj.leaf = edge.child;
				++traj.depth;
			}

			if (traj.leaf->expanded && traj.leaf->terminal) {
				traj.value = traj.leaf->terminal_value;
				trajectories.push_back(std::move(traj));
				continue;
			}

			if (traj.leaf->pending) {
				for (Edge* edge : traj.path)
					--edge->in_flight;
				break;
			}

			traj.legal = traj.position.legal_moves();
			const bool draw_by_rule = traj.depth > 0 &&
				(traj.position.is_insufficient_material() ||
				 traj.position.halfmove_clock() >= 100 ||
				 traj.position.repetition_count() >= 3);

			if (traj.legal.empty() || draw_by_rule) {
				traj.leaf->expanded = true;
				traj.leaf->terminal = true;
				if (traj.legal.empty() && traj.position.in_check())
					traj.leaf->terminal_value = -1.0;
				else
					traj.leaf->terminal_value = 0.0;
				traj.value = traj.leaf->terminal_value;
				trajectories.push_back(std::move(traj));
				continue;
			}

			traj.leaf->pending = true;
			traj.needs_eval = true;
			trajectories.push_back(std::move(traj));
		}

		if (trajectories.empty())
			break;

		std::vector<BatchQuery> queries;
		std::vector<std::size_t> eval_indices;
		queries.reserve(trajectories.size());
		eval_indices.reserve(trajectories.size());
		for (std::size_t i = 0; i < trajectories.size(); ++i) {
			if (trajectories[i].needs_eval) {
				queries.push_back({&trajectories[i].position, &trajectories[i].legal});
				eval_indices.push_back(i);
			}
		}

		if (!queries.empty()) {
			const auto eval_start = std::chrono::steady_clock::now();
			std::vector<Evaluation> evaluations;
			try {
				evaluations = evaluator_->evaluate_batch(queries);
			} catch (...) {
				for (Trajectory& traj : trajectories) {
					if (traj.needs_eval)
						traj.leaf->pending = false;
					for (Edge* edge : traj.path)
						--edge->in_flight;
				}
				throw;
			}
			const auto eval_end = std::chrono::steady_clock::now();
			eval_time_ns_ += static_cast<std::uint64_t>(
				std::max<std::int64_t>(
					0,
					std::chrono::duration_cast<std::chrono::nanoseconds>(
						eval_end - eval_start).count()));
			++batches_;
			batch_leaves_ += static_cast<int>(queries.size());

			if (evaluations.size() != queries.size()) {
				for (Trajectory& traj : trajectories) {
					if (traj.needs_eval)
						traj.leaf->pending = false;
					for (Edge* edge : traj.path)
						--edge->in_flight;
				}
				throw std::logic_error("evaluate_batch returned mismatched result count");
			}

			for (std::size_t k = 0; k < eval_indices.size(); ++k) {
				Trajectory& traj = trajectories[eval_indices[k]];
				const Evaluation& eval = evaluations[k];
				const double clamped = std::isfinite(eval.value)
					? std::clamp(eval.value, -1.0, 1.0)
					: 0.0;
				traj.leaf->initial_value = clamped;
				expand_node(*traj.leaf, traj.legal, eval.policy);
				traj.value = clamped;
				if (traj.leaf == root_node)
					root_value_ = clamped;
			}
		}

		for (Trajectory& traj : trajectories) {
			double value = traj.value;
			for (auto it = traj.path.rbegin(); it != traj.path.rend(); ++it) {
				Edge* edge = *it;
				value = -value;
				--edge->in_flight;
				++edge->visit_count;
				edge->value_sum += value;
			}
			++nodes_;
			max_depth_ = std::max(max_depth_, traj.depth);
		}

		if (on_info && config_.info_interval > 0) {
			const int bucket = nodes_ / config_.info_interval;
			if (bucket > last_reported_bucket) {
				last_reported_bucket = bucket;
				on_info(gather_info(*root_node, start_time));
			}
		}
	}

	if (!root_node->expanded) {
		const auto legal = root.legal_moves();
		const auto eval = evaluator_->evaluate_batch({{&root, &legal}});
		root_value_ = std::clamp(eval.front().value, -1.0, 1.0);
		root_node->initial_value = root_value_;
		expand_node(*root_node, legal, eval.front().policy);
	}

	const SearchInfo final_info = gather_info(*root_node, start_time);
	Move best_move = root_node->edges.front().move;
	if (!final_info.pv.empty()) {
		best_move = final_info.pv.front();
	} else {
		const Edge* best = &root_node->edges.front();
		for (const Edge& edge : root_node->edges) {
			if (edge.prior > best->prior)
				best = &edge;
		}
		best_move = best->move;
	}
	return {best_move, final_info};
}

} // namespace inez
