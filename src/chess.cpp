// SPDX-License-Identifier: GPL-3.0-only
#include "inez/chess.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace inez {
namespace {

using Board = std::array<Piece, 64>;

constexpr std::uint8_t white_kingside = 1;
constexpr std::uint8_t white_queenside = 2;
constexpr std::uint8_t black_kingside = 4;
constexpr std::uint8_t black_queenside = 8;
constexpr std::string_view start_fen =
	"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

Color other(Color color)
{
	return color == Color::White ? Color::Black : Color::White;
}

bool on_board(int file, int rank)
{
	return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

int square(int file, int rank)
{
	return rank * 8 + file;
}

bool is_piece(Piece piece, PieceType type, Color color)
{
	return piece.type == type && piece.color == color;
}

int king_square(const Board &board, Color color)
{
	for (int sq = 0; sq < 64; ++sq) {
		if (is_piece(board[sq], PieceType::King, color))
			return sq;
	}
	throw std::logic_error("position has no king");
}

bool attacked(const Board &board, int target, Color by)
{
	const int file = target % 8;
	const int rank = target / 8;
	const int pawn_rank = rank + (by == Color::White ? -1 : 1);
	for (int df : {-1, 1}) {
		if (on_board(file + df, pawn_rank) &&
		    is_piece(board[square(file + df, pawn_rank)], PieceType::Pawn, by))
			return true;
	}

	constexpr std::array<std::pair<int, int>, 8> knight_steps{{
		{1, 2}, {2, 1}, {2, -1}, {1, -2},
		{-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
	}};
	for (const auto &[df, dr] : knight_steps) {
		if (on_board(file + df, rank + dr) &&
		    is_piece(board[square(file + df, rank + dr)], PieceType::Knight, by))
			return true;
	}

	for (int df = -1; df <= 1; ++df) {
		for (int dr = -1; dr <= 1; ++dr) {
			if ((df != 0 || dr != 0) && on_board(file + df, rank + dr) &&
			    is_piece(board[square(file + df, rank + dr)], PieceType::King, by))
				return true;
		}
	}

	constexpr std::array<std::pair<int, int>, 8> rays{{
		{1, 0}, {-1, 0}, {0, 1}, {0, -1},
		{1, 1}, {1, -1}, {-1, 1}, {-1, -1},
	}};
	for (const auto &[df, dr] : rays) {
		int f = file + df;
		int r = rank + dr;
		while (on_board(f, r)) {
			const Piece piece = board[square(f, r)];
			if (piece.type != PieceType::None) {
				if (piece.color == by) {
					const bool diagonal = df != 0 && dr != 0;
					if (piece.type == PieceType::Queen ||
					    (diagonal && piece.type == PieceType::Bishop) ||
					    (!diagonal && piece.type == PieceType::Rook))
						return true;
				}
				break;
			}
			f += df;
			r += dr;
		}
	}
	return false;
}

// Only board occupancy changes here. Metadata and history are handled by
// Position::apply_unchecked after the caller has established move legality.
void apply_board(Board &board, Move move, int ep, Color mover)
{
	Piece piece = board[move.from];
	if (piece.type == PieceType::Pawn && move.to == ep &&
	    board[move.to].type == PieceType::None && move.from % 8 != move.to % 8)
		board[move.to + (mover == Color::White ? -8 : 8)] = {};

	if (piece.type == PieceType::King &&
	    (move.to - move.from == 2 || move.to - move.from == -2)) {
		const int rook_from = move.to > move.from ? move.from + 3 : move.from - 4;
		const int rook_to = move.to > move.from ? move.from + 1 : move.from - 1;
		board[rook_to] = board[rook_from];
		board[rook_from] = {};
	}

	if (move.promotion != PieceType::None)
		piece.type = move.promotion;
	board[move.to] = piece;
	board[move.from] = {};
}

char piece_letter(PieceType type)
{
	switch (type) {
	case PieceType::Pawn: return 'p';
	case PieceType::Knight: return 'n';
	case PieceType::Bishop: return 'b';
	case PieceType::Rook: return 'r';
	case PieceType::Queen: return 'q';
	case PieceType::King: return 'k';
	case PieceType::None: break;
	}
	throw std::invalid_argument("invalid piece type");
}

PieceType parse_piece(char letter)
{
	switch (letter) {
	case 'p': return PieceType::Pawn;
	case 'n': return PieceType::Knight;
	case 'b': return PieceType::Bishop;
	case 'r': return PieceType::Rook;
	case 'q': return PieceType::Queen;
	case 'k': return PieceType::King;
	default: throw std::invalid_argument("invalid FEN piece");
	}
}

std::uint64_t parse_counter(std::string_view text, bool positive)
{
	if (text.empty() || !std::all_of(text.begin(), text.end(), [](char c) {
		return c >= '0' && c <= '9';
	}))
		throw std::invalid_argument("invalid FEN move counter");
	std::uint64_t value = 0;
	const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
	if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
	    (positive && value == 0))
		throw std::invalid_argument("invalid FEN move counter");
	return value;
}

bool valid_square(int sq)
{
	return sq >= 0 && sq < 64;
}

int parse_square(std::string_view text)
{
	if (text.size() != 2 || text[0] < 'a' || text[0] > 'h' ||
	    text[1] < '1' || text[1] > '8')
		throw std::invalid_argument("invalid square");
	return square(text[0] - 'a', text[1] - '1');
}

std::string square_name(int sq)
{
	if (!valid_square(sq))
		throw std::invalid_argument("invalid square");
	std::string result;
	result += static_cast<char>('a' + sq % 8);
	result += static_cast<char>('1' + sq / 8);
	return result;
}

} // namespace

Move Move::from_uci(std::string_view text)
{
	if (text.size() != 4 && text.size() != 5)
		throw std::invalid_argument("UCI move must contain four or five characters");
	Move move{parse_square(text.substr(0, 2)), parse_square(text.substr(2, 2))};
	if (move.from == move.to)
		throw std::invalid_argument("UCI move must change square");
	if (text.size() == 5) {
		switch (text[4]) {
		case 'n': move.promotion = PieceType::Knight; break;
		case 'b': move.promotion = PieceType::Bishop; break;
		case 'r': move.promotion = PieceType::Rook; break;
		case 'q': move.promotion = PieceType::Queen; break;
		default: throw std::invalid_argument("invalid UCI promotion");
		}
	}
	return move;
}

std::string Move::uci() const
{
	if (!valid_square(from) || !valid_square(to) || from == to)
		throw std::invalid_argument("invalid move squares");
	std::string result = square_name(from) + square_name(to);
	if (promotion != PieceType::None) {
		if (promotion != PieceType::Knight && promotion != PieceType::Bishop &&
		    promotion != PieceType::Rook && promotion != PieceType::Queen)
			throw std::invalid_argument("invalid promotion type");
		result += piece_letter(promotion);
	}
	return result;
}

Position::Position()
	: Position(from_fen(start_fen))
{
}

Position Position::start()
{
	return from_fen(start_fen);
}

Position Position::from_fen(std::string_view text)
{
	std::istringstream input{std::string(text)};
	std::array<std::string, 6> fields;
	for (std::string &field : fields) {
		if (!(input >> field))
			throw std::invalid_argument("FEN must contain six fields");
	}
	std::string extra;
	if (input >> extra)
		throw std::invalid_argument("FEN must contain six fields");

	Position position(EmptyTag{});
	int rank = 7;
	int file = 0;
	bool previous_digit = false;
	for (char c : fields[0]) {
		if (c == '/') {
			if (file != 8 || rank == 0)
				throw std::invalid_argument("invalid FEN rank separator");
			--rank;
			file = 0;
			previous_digit = false;
		} else if (c >= '1' && c <= '8') {
			if (previous_digit || file + c - '0' > 8)
				throw std::invalid_argument("invalid FEN empty-square run");
			file += c - '0';
			previous_digit = true;
		} else {
			if (file >= 8)
				throw std::invalid_argument("too many squares in FEN rank");
			const auto uc = static_cast<unsigned char>(c);
			const PieceType type = parse_piece(static_cast<char>(std::tolower(uc)));
			position.board_[square(file, rank)] =
				{type, std::isupper(uc) ? Color::White : Color::Black};
			++file;
			previous_digit = false;
		}
	}
	if (rank != 0 || file != 8)
		throw std::invalid_argument("FEN board must contain eight complete ranks");

	if (fields[1] == "w")
		position.side_ = Color::White;
	else if (fields[1] == "b")
		position.side_ = Color::Black;
	else
		throw std::invalid_argument("invalid FEN active color");

	if (fields[2] != "-") {
		int previous = -1;
		for (char c : fields[2]) {
			int index = -1;
			switch (c) {
			case 'K': index = 0; break;
			case 'Q': index = 1; break;
			case 'k': index = 2; break;
			case 'q': index = 3; break;
			default: throw std::invalid_argument("invalid FEN castling rights");
			}
			if (index <= previous)
				throw std::invalid_argument("FEN castling rights must be unique and ordered KQkq");
			previous = index;
			position.castling_ |= static_cast<std::uint8_t>(1U << index);
		}
	}

	if (fields[3] != "-")
		position.ep_ = parse_square(fields[3]);
	position.halfmove_ = parse_counter(fields[4], false);
	position.fullmove_ = parse_counter(fields[5], true);

	std::array<int, 2> kings{};
	std::array<int, 2> pawns{};
	std::array<int, 2> pieces{};
	for (int sq = 0; sq < 64; ++sq) {
		const Piece piece = position.board_[sq];
		if (piece.type == PieceType::None)
			continue;
		const int color = piece.color == Color::White ? 0 : 1;
		++pieces[color];
		if (piece.type == PieceType::King)
			++kings[color];
		if (piece.type == PieceType::Pawn) {
			++pawns[color];
			if (sq / 8 == 0 || sq / 8 == 7)
				throw std::invalid_argument("FEN pawn on first or eighth rank");
		}
	}
	for (int color = 0; color < 2; ++color) {
		if (kings[color] != 1 || pawns[color] > 8 || pieces[color] > 16)
			throw std::invalid_argument("invalid FEN piece counts");
	}

	auto require_castling = [&](std::uint8_t right, Color color, int king, int rook) {
		if ((position.castling_ & right) != 0 &&
		    (!is_piece(position.board_[king], PieceType::King, color) ||
		     !is_piece(position.board_[rook], PieceType::Rook, color)))
			throw std::invalid_argument("FEN castling right lacks home king or rook");
	};
	require_castling(white_kingside, Color::White, 4, 7);
	require_castling(white_queenside, Color::White, 4, 0);
	require_castling(black_kingside, Color::Black, 60, 63);
	require_castling(black_queenside, Color::Black, 60, 56);

	if (position.ep_ != -1) {
		const bool white_turn = position.side_ == Color::White;
		const int captured = position.ep_ + (white_turn ? -8 : 8);
		const int source = position.ep_ + (white_turn ? 8 : -8);
		if (position.ep_ / 8 != (white_turn ? 5 : 2) ||
		    position.board_[position.ep_].type != PieceType::None ||
		    !is_piece(position.board_[captured], PieceType::Pawn, other(position.side_)) ||
		    position.board_[source].type != PieceType::None || position.halfmove_ != 0)
			throw std::invalid_argument("invalid FEN en-passant state");
	}

	// Includes adjacent kings and positions in which the preceding move left
	// the player who made it in check. More subtle reachability is not solved.
	if (position.in_check(other(position.side_)))
		throw std::invalid_argument("FEN side not to move is in check");
	position.history_.push_back(position.repetition_key());
	return position;
}

std::string Position::fen() const
{
	std::string result;
	for (int rank = 7; rank >= 0; --rank) {
		int empty = 0;
		for (int file = 0; file < 8; ++file) {
			const Piece piece = board_[square(file, rank)];
			if (piece.type == PieceType::None) {
				++empty;
				continue;
			}
			if (empty != 0) {
				result += static_cast<char>('0' + empty);
				empty = 0;
			}
			char c = piece_letter(piece.type);
			if (piece.color == Color::White)
				c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
			result += c;
		}
		if (empty != 0)
			result += static_cast<char>('0' + empty);
		if (rank != 0)
			result += '/';
	}
	result += side_ == Color::White ? " w " : " b ";
	if (castling_ == 0) {
		result += '-';
	} else {
		if ((castling_ & white_kingside) != 0) result += 'K';
		if ((castling_ & white_queenside) != 0) result += 'Q';
		if ((castling_ & black_kingside) != 0) result += 'k';
		if ((castling_ & black_queenside) != 0) result += 'q';
	}
	result += ' ';
	result += ep_ == -1 ? "-" : square_name(ep_);
	result += ' ';
	result += std::to_string(halfmove_);
	result += ' ';
	result += std::to_string(fullmove_);
	return result;
}

std::vector<Move> Position::pseudo_legal_moves() const
{
	std::vector<Move> moves;
	moves.reserve(64);
	auto add_pawn = [&](int from, int to) {
		if (to / 8 == 0 || to / 8 == 7) {
			for (PieceType promotion : {PieceType::Queen, PieceType::Rook,
						    PieceType::Bishop, PieceType::Knight})
				moves.push_back({from, to, promotion});
		} else {
			moves.push_back({from, to});
		}
	};
	auto can_land = [&](int target) {
		return board_[target].type == PieceType::None ||
		       (board_[target].color != side_ && board_[target].type != PieceType::King);
	};

	for (int from = 0; from < 64; ++from) {
		const Piece piece = board_[from];
		if (piece.type == PieceType::None || piece.color != side_)
			continue;
		const int file = from % 8;
		const int rank = from / 8;
		if (piece.type == PieceType::Pawn) {
			const int dr = side_ == Color::White ? 1 : -1;
			const int next_rank = rank + dr;
			if (!on_board(file, next_rank))
				continue;
			const int next = square(file, next_rank);
			if (board_[next].type == PieceType::None) {
				add_pawn(from, next);
				if (rank == (side_ == Color::White ? 1 : 6) &&
				    board_[square(file, rank + 2 * dr)].type == PieceType::None)
					moves.push_back({from, square(file, rank + 2 * dr)});
			}
			for (int df : {-1, 1}) {
				if (!on_board(file + df, next_rank))
					continue;
				const int to = square(file + df, next_rank);
				if (board_[to].type != PieceType::None &&
				    board_[to].color != side_ && board_[to].type != PieceType::King)
					add_pawn(from, to);
				else if (to == ep_ && board_[to].type == PieceType::None &&
					 is_piece(board_[square(file + df, rank)], PieceType::Pawn, other(side_)))
					moves.push_back({from, to});
			}
		} else if (piece.type == PieceType::Knight) {
			constexpr std::array<std::pair<int, int>, 8> steps{{
				{1, 2}, {2, 1}, {2, -1}, {1, -2},
				{-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
			}};
			for (const auto &[df, dr] : steps) {
				if (on_board(file + df, rank + dr)) {
					const int to = square(file + df, rank + dr);
					if (can_land(to))
						moves.push_back({from, to});
				}
			}
		} else if (piece.type == PieceType::King) {
			for (int df = -1; df <= 1; ++df) {
				for (int dr = -1; dr <= 1; ++dr) {
					if ((df != 0 || dr != 0) && on_board(file + df, rank + dr)) {
						const int to = square(file + df, rank + dr);
						if (can_land(to))
							moves.push_back({from, to});
					}
				}
			}
			const int home = side_ == Color::White ? 4 : 60;
			const std::uint8_t kingside = side_ == Color::White ? white_kingside : black_kingside;
			const std::uint8_t queenside = side_ == Color::White ? white_queenside : black_queenside;
			if (from == home && !attacked(board_, home, other(side_))) {
				if ((castling_ & kingside) != 0 &&
				    is_piece(board_[home + 3], PieceType::Rook, side_) &&
				    board_[home + 1].type == PieceType::None &&
				    board_[home + 2].type == PieceType::None &&
				    !attacked(board_, home + 1, other(side_)))
					moves.push_back({from, home + 2});
				if ((castling_ & queenside) != 0 &&
				    is_piece(board_[home - 4], PieceType::Rook, side_) &&
				    board_[home - 1].type == PieceType::None &&
				    board_[home - 2].type == PieceType::None &&
				    board_[home - 3].type == PieceType::None &&
				    !attacked(board_, home - 1, other(side_)))
					moves.push_back({from, home - 2});
			}
		} else {
			constexpr std::array<std::pair<int, int>, 8> rays{{
				{1, 0}, {-1, 0}, {0, 1}, {0, -1},
				{1, 1}, {1, -1}, {-1, 1}, {-1, -1},
			}};
			for (const auto &[df, dr] : rays) {
				const bool diagonal = df != 0 && dr != 0;
				if ((piece.type == PieceType::Bishop && !diagonal) ||
				    (piece.type == PieceType::Rook && diagonal))
					continue;
				int f = file + df;
				int r = rank + dr;
				while (on_board(f, r)) {
					const int to = square(f, r);
					if (can_land(to))
						moves.push_back({from, to});
					if (board_[to].type != PieceType::None)
						break;
					f += df;
					r += dr;
				}
			}
		}
	}
	return moves;
}

std::vector<Move> Position::legal_moves() const
{
	std::vector<Move> result;
	const auto pseudo = pseudo_legal_moves();
	result.reserve(pseudo.size());
	for (Move move : pseudo) {
		Board next = board_;
		apply_board(next, move, ep_, side_);
		if (!attacked(next, king_square(next, side_), other(side_)))
			result.push_back(move);
	}
	return result;
}

void Position::apply_unchecked(Move move)
{
	const Piece moving = board_[move.from];
	const bool capture = board_[move.to].type != PieceType::None ||
		(moving.type == PieceType::Pawn && move.to == ep_ && move.from % 8 != move.to % 8);
	const bool reset_clock = moving.type == PieceType::Pawn || capture;
	if ((!reset_clock && halfmove_ == std::numeric_limits<std::uint64_t>::max()) ||
	    (side_ == Color::Black && fullmove_ == std::numeric_limits<std::uint64_t>::max()))
		throw std::overflow_error("FEN move counter would overflow");

	// Allocate before mutation so allocation failure also preserves the state.
	if (undo_.size() == undo_.capacity())
		undo_.reserve(undo_.empty() ? 8 : undo_.size() * 2);
	if (history_.size() == history_.capacity())
		history_.reserve(history_.empty() ? 8 : history_.size() * 2);
	undo_.push_back({board_, side_, castling_, ep_, halfmove_, fullmove_});

	// Rook home-square rights are lost on departure or capture; king rights
	// are lost on every king move and cannot be recovered by returning home.
	if (moving.type == PieceType::King) {
		if (side_ == Color::White)
			castling_ &= static_cast<std::uint8_t>(~(white_kingside | white_queenside));
		else
			castling_ &= static_cast<std::uint8_t>(~(black_kingside | black_queenside));
	}
	auto clear_rook_right = [&](int sq) {
		switch (sq) {
		case 0: castling_ &= static_cast<std::uint8_t>(~white_queenside); break;
		case 7: castling_ &= static_cast<std::uint8_t>(~white_kingside); break;
		case 56: castling_ &= static_cast<std::uint8_t>(~black_queenside); break;
		case 63: castling_ &= static_cast<std::uint8_t>(~black_kingside); break;
		default: break;
		}
	};
	clear_rook_right(move.from);
	clear_rook_right(move.to);
	apply_board(board_, move, ep_, side_);
	ep_ = -1;
	if (moving.type == PieceType::Pawn &&
	    (move.to - move.from == 16 || move.to - move.from == -16))
		ep_ = (move.from + move.to) / 2;
	halfmove_ = reset_clock ? 0 : halfmove_ + 1;
	if (side_ == Color::Black)
		++fullmove_;
	side_ = other(side_);
	history_.push_back(repetition_key());
}

void Position::make_move(Move move)
{
	const auto moves = legal_moves();
	if (std::find(moves.begin(), moves.end(), move) == moves.end())
		throw std::invalid_argument("illegal chess move");
	apply_unchecked(move);
}

void Position::play_uci(std::string_view text)
{
	make_move(Move::from_uci(text));
}

void Position::unmake_move()
{
	if (undo_.empty())
		throw std::logic_error("no move to unmake");
	const Undo previous = undo_.back();
	board_ = previous.board;
	side_ = previous.side;
	castling_ = previous.castling;
	ep_ = previous.ep;
	halfmove_ = previous.halfmove;
	fullmove_ = previous.fullmove;
	undo_.pop_back();
	history_.pop_back();
}

bool Position::in_check() const
{
	return in_check(side_);
}

bool Position::in_check(Color color) const
{
	return attacked(board_, king_square(board_, color), other(color));
}

Piece Position::piece_at(int sq) const
{
	if (!valid_square(sq))
		throw std::out_of_range("invalid board square");
	return board_[sq];
}

bool Position::has_legal_en_passant() const
{
	if (ep_ == -1)
		return false;
	const int from_rank = ep_ / 8 + (side_ == Color::White ? -1 : 1);
	for (int df : {-1, 1}) {
		const int from_file = ep_ % 8 + df;
		if (!on_board(from_file, from_rank))
			continue;
		const int from = square(from_file, from_rank);
		if (!is_piece(board_[from], PieceType::Pawn, side_))
			continue;
		Board next = board_;
		apply_board(next, {from, ep_}, ep_, side_);
		if (!attacked(next, king_square(next, side_), other(side_)))
			return true;
	}
	return false;
}

RepetitionKey Position::repetition_key() const
{
	return {board_, side_, castling_, has_legal_en_passant() ? ep_ : -1};
}

std::uint64_t Position::repetition_hash() const
{
	const RepetitionKey key = repetition_key();
	std::uint64_t hash = 14695981039346656037ULL;
	auto append = [&](unsigned int byte) {
		hash ^= byte;
		hash *= 1099511628211ULL;
	};
	for (const Piece piece : key.board) {
		const unsigned int type = static_cast<unsigned int>(piece.type);
		append(type == 0 ? 0 : type + (piece.color == Color::Black ? 8U : 0U));
	}
	append(key.side_to_move == Color::Black ? 1U : 0U);
	append(key.castling_rights);
	append(static_cast<unsigned int>(key.en_passant_square + 1));
	return hash;
}

int Position::repetition_count() const
{
	const RepetitionKey key = repetition_key();
	return static_cast<int>(std::count(history_.begin(), history_.end(), key));
}

SearchKey Position::search_key() const
{
	return {repetition_key(), halfmove_, history_};
}

bool Position::is_insufficient_material() const
{
	int knights = 0;
	int bishops = 0;
	int bishop_color = -1;
	bool all_bishops_same_color = true;
	for (int sq = 0; sq < 64; ++sq) {
		switch (board_[sq].type) {
		case PieceType::Pawn:
		case PieceType::Rook:
		case PieceType::Queen:
			return false;
		case PieceType::Knight:
			++knights;
			break;
		case PieceType::Bishop: {
			++bishops;
			const int color = (sq % 8 + sq / 8) % 2;
			if (bishop_color != -1 && bishop_color != color)
				all_bishops_same_color = false;
			bishop_color = color;
			break;
		}
		case PieceType::None:
		case PieceType::King:
			break;
		}
	}
	return knights + bishops <= 1 ||
	       (knights == 0 && all_bishops_same_color);
}

bool Position::can_claim_fifty_moves() const
{
	if (halfmove_ < 99)
		return false;
	const auto moves = legal_moves();
	if (halfmove_ >= 100)
		return !moves.empty();
	for (Move move : moves) {
		if (board_[move.from].type == PieceType::Pawn ||
		    board_[move.to].type != PieceType::None)
			continue;
		Position next = *this;
		// Fullmove numbering has no bearing on whether a draw is claimable.
		next.fullmove_ = 1;
		next.apply_unchecked(move);
		if (next.halfmove_ >= 100 && !next.legal_moves().empty())
			return true;
	}
	return false;
}

bool Position::can_claim_threefold_repetition() const
{
	if (repetition_count() >= 3)
		return true;
	for (Move move : legal_moves()) {
		Position next = *this;
		// Repetition identity does not include either numerical clock.
		next.halfmove_ = 0;
		next.fullmove_ = 1;
		next.apply_unchecked(move);
		if (next.repetition_count() >= 3)
			return true;
	}
	return false;
}

Outcome Position::outcome(bool claim_draw) const
{
	const auto moves = legal_moves();
	if (moves.empty()) {
		if (in_check())
			return {Termination::Checkmate, other(side_)};
		return {Termination::Stalemate, std::nullopt};
	}
	if (is_insufficient_material())
		return {Termination::InsufficientMaterial, std::nullopt};
	if (halfmove_ >= 150)
		return {Termination::SeventyFiveMoveRule, std::nullopt};
	if (repetition_count() >= 5)
		return {Termination::FivefoldRepetition, std::nullopt};
	if (claim_draw) {
		if (can_claim_fifty_moves())
			return {Termination::FiftyMoveRule, std::nullopt};
		if (can_claim_threefold_repetition())
			return {Termination::ThreefoldRepetition, std::nullopt};
	}
	return {};
}

std::uint64_t Position::perft_impl(int depth)
{
	if (depth == 0)
		return 1;
	const auto moves = legal_moves();
	if (depth == 1)
		return static_cast<std::uint64_t>(moves.size());
	std::uint64_t nodes = 0;
	for (Move move : moves) {
		apply_unchecked(move);
		const std::uint64_t count = perft_impl(depth - 1);
		unmake_move();
		if (count > std::numeric_limits<std::uint64_t>::max() - nodes)
			throw std::overflow_error("perft count would overflow");
		nodes += count;
	}
	return nodes;
}

std::uint64_t Position::perft(int depth) const
{
	if (depth < 0)
		throw std::invalid_argument("perft depth must be nonnegative");
	Position scratch = *this;
	// Perft counts move legality, independent of clocks and draw adjudication.
	scratch.halfmove_ = 0;
	scratch.fullmove_ = 1;
	scratch.undo_.clear();
	scratch.history_.assign(1, scratch.repetition_key());
	return scratch.perft_impl(depth);
}

} // namespace inez
