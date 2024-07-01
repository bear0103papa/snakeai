let model;
let game;

const GRID_SIZE = 15;
const CELL_SIZE = 20;

async function loadModel() {
    try {
        console.log('Attempting to load model...');
        model = await tf.loadLayersModel('snakeai/model.json');
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
        this.snake = [{ x: Math.floor(GRID_SIZE / 2), y: Math.floor(GRID_SIZE / 2) }];
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
        let head = { ...this.snake[0] };
        switch (this.direction) {
            case 'up': head.y--; break;
            case 'right': head.x++; break;
            case 'down': head.y++; break;
            case 'left': head.x--; break;
        }
        if (head.x < 0) head.x = GRID_SIZE - 1;
        if (head.x >= GRID_SIZE) head.x = 0;
        if (head.y < 0) head.y = GRID_SIZE - 1;
        if (head.y >= GRID_SIZE) head.y = 0;
        
        if (this.snake.some(segment => segment.x === head.x && segment.y === head.y)) {
            this.reset();
            return;
        }

        this.snake.unshift(head);

        if (head.x === this.food.x && head.y === this.food.y) {
            this.score++;
            this.food = this.generateFood();
        } else {
            this.snake.pop();
        }
    }

    getState() {
        const state = tf.zeros([GRID_SIZE, GRID_SIZE, 3]);
        for (const segment of this.snake) {
            state.buffer().set(1, segment.y, segment.x, 1);
        }
        state.buffer().set(1, this.food.y, this.food.x, 0);
        state.buffer().set(1, this.snake[0].y, this.snake[0].x, 2);
        return state;
    }

    setDirection(direction) {
        this.direction = direction;
    }
}

async function playGame() {
    game = new SnakeGame();
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = GRID_SIZE * CELL_SIZE;
    canvas.height = GRID_SIZE * CELL_SIZE;

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'red';
        ctx.fillRect(game.food.x * CELL_SIZE, game.food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE);

        for (const segment of game.snake) {
            ctx.fillStyle = 'green';
            ctx.fillRect(segment.x * CELL_SIZE, segment.y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        }

        ctx.fillStyle = 'black';
        ctx.fillText(`Score: ${game.score}`, 10, canvas.height - 10);
    }

    async function step() {
        const state = game.getState();
        const action = (await model.predict(state.expandDims(0))).argMax(-1).dataSync()[0];

        switch (action) {
            case 0: game.setDirection('up'); break;
            case 1: game.setDirection('right'); break;
            case 2: game.setDirection('down'); break;
            case 3: game.setDirection('left'); break;
        }

        game.step();
        draw();
        setTimeout(step, 100);
    }

    draw();
    setTimeout(step, 100);
}

document.getElementById('startButton').addEventListener('click', playGame);
window.addEventListener('load', loadModel);
