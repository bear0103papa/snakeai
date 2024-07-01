let model;
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const gridSize = 20;
const tileCount = canvas.width / gridSize;

let snake = [{x: 10, y: 10}];
let food = {x: 15, y: 15};
let dx = 0;
let dy = 0;

async function loadModel() {
    model = await tf.loadLayersModel('snake_ai_model.json');
}

function getState() {
    let state = new Array(tileCount).fill(0).map(() => new Array(tileCount).fill(0));
    snake.forEach(segment => {
        state[segment.y][segment.x] = 1;
    });
    state[food.y][food.x] = 2;
    return state;
}

function predictMove() {
    const state = getState();
    const tensorState = tf.tensor4d([state.map(row => row.map(cell => [cell === 1 ? 1 : 0, cell === 2 ? 1 : 0, cell === 1 && snake[0].x === cell.x && snake[0].y === cell.y ? 1 : 0]))]);
    const prediction = model.predict(tensorState);
    const move = prediction.argMax(1).dataSync()[0];
    
    switch(move) {
        case 0: dy = -1; dx = 0; break;  // 上
        case 1: dx = 1; dy = 0; break;   // 右
        case 2: dy = 1; dx = 0; break;   // 下
        case 3: dx = -1; dy = 0; break;  // 左
    }
}

function gameLoop() {
    predictMove();
    
    const head = {x: snake[0].x + dx, y: snake[0].y + dy};
    snake.unshift(head);
    
    if (head.x === food.x && head.y === food.y) {
        food = {
            x: Math.floor(Math.random() * tileCount),
            y: Math.floor(Math.random() * tileCount)
        };
    } else {
        snake.pop();
    }
    
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = 'green';
    snake.forEach(segment => {
        ctx.fillRect(segment.x * gridSize, segment.y * gridSize, gridSize - 2, gridSize - 2);
    });
    
    ctx.fillStyle = 'red';
    ctx.fillRect(food.x * gridSize, food.y * gridSize, gridSize - 2, gridSize - 2);
    
    if (head.x < 0 || head.x >= tileCount || head.y < 0 || head.y >= tileCount || snake.slice(1).some(segment => segment.x === head.x && segment.y === head.y)) {
        clearInterval(gameInterval);
        alert('Game Over!');
    }
}

loadModel().then(() => {
    gameInterval = setInterval(gameLoop, 100);
});
