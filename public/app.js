const COLS = 10;
const ROWS = 22;
const BLOCK = 30;
const COLORS = {
  I: "#00f0f0",
  J: "#2a68f7",
  L: "#f7a22a",
  O: "#f4e64a",
  S: "#44dd56",
  T: "#b95cff",
  Z: "#f24d5b",
  G: "#555d6b",
};

const SHAPES = {
  I: [[0, 1], [1, 1], [2, 1], [3, 1]],
  J: [[0, 0], [0, 1], [1, 1], [2, 1]],
  L: [[2, 0], [0, 1], [1, 1], [2, 1]],
  O: [[1, 0], [2, 0], [1, 1], [2, 1]],
  S: [[1, 0], [2, 0], [0, 1], [1, 1]],
  T: [[1, 0], [0, 1], [1, 1], [2, 1]],
  Z: [[0, 0], [1, 0], [1, 1], [2, 1]],
};

const boardCanvas = document.getElementById("board");
const boardCtx = boardCanvas.getContext("2d");
const holdCtx = document.getElementById("hold").getContext("2d");
const nextCtx = document.getElementById("next").getContext("2d");
const overlay = document.getElementById("overlay");
const scoreEl = document.getElementById("score");
const linesEl = document.getElementById("lines");
const levelEl = document.getElementById("level");
const comboEl = document.getElementById("combo");

let board;
let active;
let queue;
let hold;
let canHold;
let score;
let lines;
let combo;
let level;
let dropMs;
let lastTime;
let acc;
let running = false;
let paused = false;
let gameOver = false;

function emptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(null));
}

function shuffleBag() {
  const bag = Object.keys(SHAPES);
  for (let i = bag.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [bag[i], bag[j]] = [bag[j], bag[i]];
  }
  return bag;
}

function refillQueue() {
  while (queue.length < 7) queue.push(...shuffleBag());
}

function newPiece(type = queue.shift()) {
  refillQueue();
  return { type, x: 3, y: 0, r: 0 };
}

function cells(piece = active) {
  let pts = SHAPES[piece.type].map(([x, y]) => [x, y]);
  if (piece.type !== "O") {
    for (let n = 0; n < piece.r % 4; n++) {
      pts = pts.map(([x, y]) => [2 - y, x]);
    }
  }
  return pts.map(([x, y]) => [x + piece.x, y + piece.y]);
}

function collides(piece = active) {
  return cells(piece).some(([x, y]) => x < 0 || x >= COLS || y >= ROWS || (y >= 0 && board[y][x]));
}

function move(dx, dy) {
  const moved = { ...active, x: active.x + dx, y: active.y + dy };
  if (collides(moved)) return false;
  active = moved;
  return true;
}

function rotate(dir) {
  const rotated = { ...active, r: (active.r + dir + 4) % 4 };
  const kicks = [0, -1, 1, -2, 2];
  for (const k of kicks) {
    const test = { ...rotated, x: rotated.x + k };
    if (!collides(test)) {
      active = test;
      return;
    }
  }
}

function hardDrop() {
  let gained = 0;
  while (move(0, 1)) gained += 2;
  score += gained;
  lockPiece();
}

function lockPiece() {
  for (const [x, y] of cells()) {
    if (y < 0) {
      endGame();
      return;
    }
    board[y][x] = active.type;
  }
  clearLines();
  active = newPiece();
  canHold = true;
  if (collides(active)) endGame();
}

function clearLines() {
  let cleared = 0;
  board = board.filter((row) => {
    if (row.every(Boolean)) {
      cleared++;
      return false;
    }
    return true;
  });
  while (board.length < ROWS) board.unshift(Array(COLS).fill(null));

  if (cleared) {
    combo += 1;
    lines += cleared;
    level = Math.floor(lines / 10) + 1;
    dropMs = Math.max(85, 760 - (level - 1) * 55);
    score += [0, 100, 300, 500, 800][cleared] * level + combo * 40;
  } else {
    combo = 0;
  }
}

function holdPiece() {
  if (!canHold) return;
  const current = active.type;
  active = hold ? newPiece(hold) : newPiece();
  hold = current;
  canHold = false;
}

function ghostY() {
  const ghost = { ...active };
  while (!collides({ ...ghost, y: ghost.y + 1 })) ghost.y++;
  return ghost.y;
}

function drawCell(ctx, x, y, color, size = BLOCK, alpha = 1) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.fillRect(x * size + 1, y * size + 1, size - 2, size - 2);
  ctx.fillStyle = "rgba(255,255,255,.16)";
  ctx.fillRect(x * size + 3, y * size + 3, size - 6, 4);
  ctx.globalAlpha = 1;
}

function drawMini(ctx, type, ox, oy) {
  if (!type) return;
  const size = 22;
  const shape = SHAPES[type];
  for (const [x, y] of shape) drawCell(ctx, ox + x, oy + y, COLORS[type], size);
}

function draw() {
  boardCtx.fillStyle = "#050608";
  boardCtx.fillRect(0, 0, boardCanvas.width, boardCanvas.height);
  boardCtx.strokeStyle = "#242b36";
  for (let x = 0; x <= COLS; x++) {
    boardCtx.beginPath();
    boardCtx.moveTo(x * BLOCK, 0);
    boardCtx.lineTo(x * BLOCK, ROWS * BLOCK);
    boardCtx.stroke();
  }
  for (let y = 0; y <= ROWS; y++) {
    boardCtx.beginPath();
    boardCtx.moveTo(0, y * BLOCK);
    boardCtx.lineTo(COLS * BLOCK, y * BLOCK);
    boardCtx.stroke();
  }

  board.forEach((row, y) => row.forEach((type, x) => {
    if (type) drawCell(boardCtx, x, y, COLORS[type]);
  }));

  if (active) {
    const ghost = { ...active, y: ghostY() };
    cells(ghost).forEach(([x, y]) => y >= 0 && drawCell(boardCtx, x, y, COLORS[active.type], BLOCK, 0.22));
    cells(active).forEach(([x, y]) => y >= 0 && drawCell(boardCtx, x, y, COLORS[active.type]));
  }

  holdCtx.clearRect(0, 0, 112, 84);
  drawMini(holdCtx, hold, 1, 1);
  nextCtx.clearRect(0, 0, 120, 360);
  queue.slice(0, 5).forEach((type, i) => drawMini(nextCtx, type, 1, i * 3 + 1));

  scoreEl.textContent = score.toLocaleString();
  linesEl.textContent = lines;
  levelEl.textContent = level;
  comboEl.textContent = combo;
}

function tick(time = 0) {
  if (!running) return;
  const dt = time - lastTime;
  lastTime = time;
  if (!paused && !gameOver) {
    acc += dt;
    if (acc >= dropMs) {
      if (!move(0, 1)) lockPiece();
      acc = 0;
    }
  }
  draw();
  requestAnimationFrame(tick);
}

function startGame() {
  board = emptyBoard();
  queue = [];
  refillQueue();
  hold = null;
  canHold = true;
  score = 0;
  lines = 0;
  combo = 0;
  level = 1;
  dropMs = 760;
  active = newPiece();
  acc = 0;
  lastTime = performance.now();
  gameOver = false;
  paused = false;
  running = true;
  overlay.classList.add("hidden");
  requestAnimationFrame(tick);
}

function endGame() {
  gameOver = true;
  overlay.querySelector("h1").textContent = "Game Over";
  overlay.querySelector("p").textContent = "Press Enter or tap Restart";
  overlay.querySelector("button").textContent = "Restart";
  overlay.classList.remove("hidden");
}

function togglePause() {
  if (!running || gameOver) return;
  paused = !paused;
  overlay.querySelector("h1").textContent = paused ? "Paused" : "Tetris";
  overlay.querySelector("p").textContent = "Press Enter to continue";
  overlay.querySelector("button").textContent = "Resume";
  overlay.classList.toggle("hidden", !paused);
}

function handleKey(key) {
  if (key === "Enter") {
    if (!running || gameOver) startGame();
    else if (paused) togglePause();
    return;
  }
  if (!running || paused || gameOver) return;
  if (key === "f" || key === "ArrowLeft") move(-1, 0);
  if (key === "j" || key === "ArrowRight") move(1, 0);
  if (key === "q" || key === "p" || key === "ArrowDown") {
    if (move(0, 1)) score += 1;
  }
  if (key === "i" || key === "ArrowUp") rotate(1);
  if (key === "e" || key === "z") rotate(-1);
  if (key === "o" || key === "a") rotate(2);
  if (key === " ") hardDrop();
  if (key === "w" || key === "c") holdPiece();
  if (key === "Escape") togglePause();
}

document.addEventListener("keydown", (event) => {
  if (event.key === " ") event.preventDefault();
  handleKey(event.key);
});

document.getElementById("start").addEventListener("click", () => {
  if (!running || gameOver) startGame();
  else if (paused) togglePause();
});
document.getElementById("pause").addEventListener("click", togglePause);
document.getElementById("restart").addEventListener("click", startGame);
document.querySelectorAll(".touch button").forEach((button) => {
  button.addEventListener("pointerdown", () => handleKey(button.dataset.key));
});

board = emptyBoard();
queue = [];
refillQueue();
active = newPiece();
hold = null;
score = 0;
lines = 0;
combo = 0;
level = 1;
draw();

