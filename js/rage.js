/**
 * Rage Chess Engine — JavaScript Wrapper
 *
 * Pure ES module, zero dependencies. Wraps the wasm-bindgen exports
 * from rage-wasm into a clean, ergonomic API.
 *
 * Usage:
 *   import { Rage } from './rage.js';
 *   const rage = await Rage.initialize('./pkg/rage_wasm.js');
 *
 *   // Chess library (no search):
 *   const board = rage.createBoard();
 *   board.makeMove('e2e4');
 *   console.log(board.legalMoves());
 *
 *   // Engine with search:
 *   const engine = rage.createEngine({ hashMb: 16 });
 *   await engine.loadWeights('./models/lite.nnue');
 *   engine.onSearchInfo(info => console.log(`depth ${info.depth} score ${info.score}`));
 *   const result = engine.search({ depth: 10 });
 *   console.log(result.bestMove, result.score);
 */

/**
 * Main entry point. Call `Rage.initialize()` to load the wasm module.
 */
export class Rage {
  /** @type {object} raw wasm-bindgen module */
  #wasm;

  constructor(wasm) {
    this.#wasm = wasm;
  }

  /**
   * Load and initialize the wasm module.
   * @param {string} wasmModuleUrl - URL to the wasm-bindgen JS glue file (e.g. './pkg/rage_wasm.js')
   * @returns {Promise<Rage>}
   */
  static async initialize(wasmModuleUrl) {
    const mod = await import(wasmModuleUrl);
    await mod.default();
    return new Rage(mod);
  }

  /**
   * Create a chess board (no engine/search).
   * @param {string} [fen] - Optional FEN string. Defaults to starting position.
   * @returns {Board}
   */
  createBoard(fen) {
    const inner = fen
      ? this.#wasm.RageBoard.fromFen(fen)
      : new this.#wasm.RageBoard();
    return new Board(inner, this.#wasm);
  }

  /**
   * Create a search-capable engine.
   * @param {{ hashMb?: number }} [opts]
   * @returns {Engine}
   */
  createEngine(opts) {
    const hashMb = opts?.hashMb ?? 16;
    const inner = new this.#wasm.RageEngine(hashMb);
    return new Engine(inner, this.#wasm);
  }
}

/**
 * Chess board — move generation, validation, and game state.
 */
export class Board {
  /** @type {object} wasm RageBoard instance */
  #inner;
  #wasm;

  constructor(inner, wasm) {
    this.#inner = inner;
    this.#wasm = wasm;
  }

  /** FEN string of the current position. */
  get fen() {
    return this.#inner.fen();
  }

  /** ASCII representation of the board. */
  toString() {
    return this.#inner.ascii();
  }

  /** Side to move: 'w' or 'b'. */
  get sideToMove() {
    return this.#inner.sideToMove();
  }

  /** Whether the side to move is in check. */
  get inCheck() {
    return this.#inner.inCheck();
  }

  /** Halfmove clock (for 50-move rule). */
  get halfmoveClock() {
    return this.#inner.halfmoveClock();
  }

  /** Fullmove number. */
  get fullmoveNumber() {
    return this.#inner.fullmoveNumber();
  }

  /** Castling rights as a string (e.g. "KQkq", "Kq", "-"). */
  get castlingRights() {
    return this.#inner.castlingRights();
  }

  /** Number of half-moves (plies) played from the starting position. */
  get plyCount() {
    return this.#inner.plyCount();
  }

  /**
   * Get legal moves as an array of UCI strings.
   * @returns {string[]}
   */
  legalMoves() {
    return this.#inner.legalMoves();
  }

  /**
   * Play a move in UCI notation (e.g. 'e2e4', 'e7e8q').
   * @param {string} uci
   * @returns {Board} this (chainable)
   */
  makeMove(uci) {
    this.#inner.makeMove(uci);
    return this;
  }

  /**
   * Undo the last move.
   * @returns {Board} this (chainable)
   */
  undoMove() {
    this.#inner.undoMove();
    return this;
  }

  /**
   * Check if a UCI move is legal in the current position.
   * @param {string} uci
   * @returns {boolean}
   */
  isLegal(uci) {
    return this.legalMoves().includes(uci);
  }

  /**
   * Check if a UCI move is a capture.
   * @param {string} uci
   * @returns {boolean}
   */
  isCapture(uci) {
    return this.#inner.isCapture(uci);
  }

  /** Whether the position is checkmate. */
  get isCheckmate() {
    return this.#inner.isCheckmate();
  }

  /** Whether the position is stalemate. */
  get isStalemate() {
    return this.#inner.isStalemate();
  }

  /** Whether the position is a draw. */
  get isDraw() {
    return this.#inner.isDraw();
  }

  /** Whether the game is over (checkmate, stalemate, or draw). */
  get isGameOver() {
    return this.#inner.isGameOver();
  }

  /**
   * Get the piece on a square.
   * @param {string} sq - Algebraic notation (e.g. 'e4')
   * @returns {string|null} Piece character ('P', 'n', 'K', etc.) or null if empty
   */
  pieceOn(sq) {
    const p = this.#inner.pieceOn(sq);
    return p === '' ? null : p;
  }

  /**
   * Run perft (performance test move generation) to the given depth.
   * @param {number} depth
   * @returns {number} Node count
   */
  perft(depth) {
    return this.#inner.perft(depth);
  }

  /**
   * Deep clone this board, preserving full undo history.
   * @returns {Board}
   */
  clone() {
    return new Board(this.#inner.clone(), this.#wasm);
  }

  /** Zobrist hash as a hex string. */
  get zobristHash() {
    return this.#inner.zobristHash();
  }

  /** Free the underlying wasm memory. */
  free() {
    this.#inner.free();
  }
}

/**
 * Chess engine — search and evaluation.
 */
export class Engine {
  /** @type {object} wasm RageEngine instance */
  #inner;
  #wasm;

  constructor(inner, wasm) {
    this.#inner = inner;
    this.#wasm = wasm;
  }

  /**
   * Set the position from a FEN string.
   * @param {string} fen
   * @returns {Engine} this (chainable)
   */
  setPosition(fen) {
    this.#inner.setPosition(fen);
    return this;
  }

  /**
   * Reset to the starting position.
   * @returns {Engine} this (chainable)
   */
  setStartPosition() {
    this.#inner.setStartPosition();
    return this;
  }

  /**
   * Play a move in UCI notation.
   * @param {string} uci
   * @returns {Engine} this (chainable)
   */
  makeMove(uci) {
    this.#inner.makeMove(uci);
    return this;
  }

  /**
   * Clear the transposition table and start a new game.
   * @returns {Engine} this (chainable)
   */
  newGame() {
    this.#inner.newGame();
    return this;
  }

  /**
   * Load NNUE weights from a URL.
   * @param {string} url - URL to a .nnue file
   * @returns {Promise<Engine>} this (chainable)
   */
  async loadWeights(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch weights: ${response.status} ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    this.#inner.loadWeights(new Uint8Array(buffer));
    return this;
  }

  /**
   * Load NNUE weights from raw bytes.
   * @param {Uint8Array} data
   * @returns {Engine} this (chainable)
   */
  loadWeightsFromBytes(data) {
    this.#inner.loadWeights(data);
    return this;
  }

  /**
   * Register a callback invoked after each completed search iteration.
   * @param {function} callback - Receives `{ depth, seldepth, score, nodes, timeMs, pv, hashfull, tbhits }`
   * @returns {Engine} this (chainable)
   */
  onSearchInfo(callback) {
    this.#inner.onSearchInfo(callback);
    return this;
  }

  /**
   * Remove the search info callback.
   * @returns {Engine} this (chainable)
   */
  clearSearchInfo() {
    this.#inner.clearSearchInfo();
    return this;
  }

  /**
   * Search the current position.
   * @param {{ depth?: number, timeMs?: number, nodes?: number }} opts
   * @returns {{ bestMove: string, score: number, depth: number, nodes: number, pv: string, isMate: boolean, mateIn: number }}
   */
  search(opts) {
    let result;
    if (opts?.depth != null) {
      result = this.#inner.searchDepth(opts.depth);
    } else if (opts?.timeMs != null) {
      result = this.#inner.searchTime(opts.timeMs);
    } else if (opts?.nodes != null) {
      result = this.#inner.searchNodes(opts.nodes);
    } else {
      result = this.#inner.searchDepth(10);
    }
    const out = {
      bestMove: result.bestMove,
      score: result.score,
      depth: result.depth,
      nodes: result.nodes,
      pv: result.pv,
      isMate: result.isMate,
      mateIn: result.mateIn,
    };
    result.free();
    return out;
  }

  /**
   * Static NNUE evaluation of the current position.
   * @returns {number|null} Centipawns from side-to-move perspective, or null if no network loaded.
   */
  evaluate() {
    return this.#inner.evaluate() ?? null;
  }

  /** FEN of the current position. */
  get fen() {
    return this.#inner.fen();
  }

  /**
   * Get legal moves as an array of UCI strings.
   * @returns {string[]}
   */
  legalMoves() {
    return this.#inner.legalMoves();
  }

  /** Whether the game is over. */
  get isGameOver() {
    return this.#inner.isGameOver();
  }

  /** Free the underlying wasm memory. */
  free() {
    this.#inner.free();
  }
}
