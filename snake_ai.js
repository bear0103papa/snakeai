let model;

async function loadModel() {
  model = await tf.loadLayersModel('web_model/model.json');
}

function startGame() {
  // 實現遊戲邏輯,使用加載的模型來決定蛇的移動
}

// 在頁面加載時加載模型
loadModel();
