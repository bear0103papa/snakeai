let model;
let game;

const GRID_SIZE = 15;
const CELL_SIZE = 20;
async function loadModel() {
    try {
        console.log('Attempting to load model...');
        model = await tf.loadLayersModel('/snakeai/model.json');
        console.log('Model loaded successfully');
        document.getElementById('startButton').disabled = false;
    } catch (error) {
        console.error('Failed to load the model:', error);
    }
}

class SnakeGame {
    constructor() {
        this.reset();
    }

    reset() {
        this.snake = [{x: Math.floor(GRID_SIZE / 2), y: Math.floor(GRID_SIZE / 2)}];
        this.food = this.generateFood();
        this.direction = 'right';
        this.score = 0;
    }

    generateFood() {
        let food;
        do {
            food = {
                x: Math.floor(Math.random() * GRID_SIZE),
                y: Math.floor(Math.random() * GRID_SIZE)
            };
        } while (this.snake.some(segment => segment.x === food.x && segment.y === food.y));
        return food;
    }

    step() {
        let head = {...this.snake[0]};
        switch(this.direction) {
            case 'up': head.y--; break;
            case 'down': head.y++; break;
            case 'left': head.x--; break;
            case 'right': head.x++; break;
        }

        if (head.x < 0) head.x = GRID_SIZE - 1;
        if (head.x >= GRID_SIZE) head.x = 0;
        if (head.y < 0) head.y = GRID_SIZE - 1;
        if (head.y >= GRID_SIZE) head.y = 0;

        this.snake.unshift(head);

        if (head.x === this.food.x && head.y === this.food.y) {
            this.score++;
            this.food = this.generateFood();
        } else {
            this.snake.pop();
        }

        return !this.checkCollision();
    }

    checkCollision() {
        const head = this.snake[0];
        return this.snake.slice(1).some(segment => segment.x === head.x && segment.y === head.y);
    }

    getState() {
        let state = new Array(GRID_SIZE).fill(0).map(() => new Array(GRID_SIZE).fill(0));
        this.snake.forEach((segment, index) => {
            state[segment.y][segment.x] = index === 0 ? 2 : 1;
        });
        state[this.food.y][this.food.x] = 3;
        return state;
    }
}

function drawGame() {
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw snake
    game.snake.forEach((segment, index) => {
        ctx.fillStyle = index === 0 ? 'darkgreen' : 'green';
        ctx.fillRect(segment.x * CELL_SIZE, segment.y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
    });

    // Draw food
    ctx.fillStyle = 'red';
    ctx.fillRect(game.food.x * CELL_SIZE, game.food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
}

async function getNextMove() {
    const state = game.getState();
    const tensor = tf.tensor4d([state.map(row => row.map(cell => [cell === 2 ? 1 : 0, cell === 1 ? 1 : 0, cell === 3 ? 1 : 0]))]);
    const prediction = await model.predict(tensor).data();
    const move = ['up', 'right', 'down', 'left'][prediction.indexOf(Math.max(...prediction))];
    tensor.dispose();
    return move;
}

async function gameLoop() {
    if (model) {
        game.direction = await getNextMove();
    }
    if (game.step()) {
        drawGame();
        document.getElementById('score').textContent = `Score: ${game.score}`;
        requestAnimationFrame(gameLoop);
    } else {
        alert(`Game Over! Score: ${game.score}`);
    }
}

async function startGame() {
    if (!model) {
        await loadModel();
    }
    game = new SnakeGame();
    gameLoop();
}

document.getElementById('startButton').addEventListener('click', startGame);
document.getElementById('startButton').disabled = true;

// Load the model when the page loads
loadModel();
