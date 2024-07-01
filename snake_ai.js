let model;
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const gridSize = 40;
const tileCount = canvas.width / gridSize;

let snake = [{x: 10, y: 10}];
let food = {x: 15, y: 15};
let dx = 0;
let dy = 0;

async function loadModel() {
    model = await tf.loadLayersModel('snake_ai_model.json');
}

function getState() {
    let state = new Array(tileCount).fill(0).map(() => new Array(tileCount).fill(0).map(() => [0, 0, 0]));
    snake.forEach((segment, index) => {
        if (index === 0) {
            state[segment.y][segment.x] = [0, 0, 1];  // 蛇頭 - 藍色
        } else {
            state[segment.y][segment.x] = [0, 1, 0];  // 蛇身 - 綠色
        }
    });
    state[food.y][food.x] = [1, 0, 0];  // 食物 - 紅色
    return state;
}

function predictMove() {
    const state = getState();
    const tensorState = tf.tensor4d([state]);
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
    
    // 繪製蛇身
    ctx.fillStyle = 'green';
    snake.slice(1).forEach(segment => {
        ctx.fillRect(segment.x * gridSize, segment.y * gridSize, gridSize - 2, gridSize - 2);
    });
    
    // 繪製蛇頭
    ctx.fillStyle = 'blue';
    ctx.fillRect(snake[0].x * gridSize, snake[0].y * gridSize, gridSize - 2, gridSize - 2);
    
    // 繪製食物
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
