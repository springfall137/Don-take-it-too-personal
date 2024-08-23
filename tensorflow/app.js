let model;
let imgElement;
const progressBar = document.getElementById('progress-bar');
const trainingProgress = document.getElementById('training-progress');

async function loadModel() {
    progressBar.style.display = 'block';
    progressBar.value = 20;

    // 모델을 불러옵니다.
    model = await tf.loadLayersModel('model.json');
    document.getElementById('result').innerText = 'Model loaded, ready to predict!';

    progressBar.value = 100;
    setTimeout(() => progressBar.style.display = 'none', 500); // 모델 로딩 후 게이지 숨기기
}

document.getElementById('image-upload').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (event) {
            imgElement = document.createElement('img');
            imgElement.src = event.target.result;
            imgElement.onload = function () {
                document.getElementById('predict-button').disabled = false; // 버튼 활성화
            }
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('predict-button').addEventListener('click', async () => {
    if (imgElement) {
        progressBar.style.display = 'block';
        progressBar.value = 50;

        // 이미지를 Tensor로 변환합니다.
        const tensor = tf.browser.fromPixels(imgElement)
            .resizeNearestNeighbor([150, 150]) // 이미지 크기 조정
            .toFloat()
            .expandDims(); // 차원 확장

        // 예측을 수행합니다.
        const predictions = await model.predict(tensor).data();
        const probability = predictions[0];
        const isDog = probability > 0.5;

        progressBar.value = 100;
        document.getElementById('result').innerText =
            `Prediction: ${isDog ? 'It\'s a Dog!' : 'It\'s a Cat!'}\nProbability: ${(probability * 100).toFixed(2)}%`;

        setTimeout(() => progressBar.style.display = 'none', 500); // 예측 후 게이지 숨기기
    }
});

// 모델 훈련 진행 상황을 추적하는 콜백 설정
async function trainModel() {
    const batchSize = 32;
    const epochs = 10;

    await model.fit(tensor, labels, {
        batchSize,
        epochs,
        callbacks: {
            onEpochBegin: (epoch, logs) => {
                trainingProgress.innerText = `Epoch ${epoch + 1} of ${epochs}`;
            },
            onEpochEnd: (epoch, logs) => {
                progressBar.value = ((epoch + 1) / epochs) * 100;
            },
            onBatchBegin: (batch, logs) => {
                trainingProgress.innerText += `\nTraining batch ${batch + 1}`;
            },
            onBatchEnd: (batch, logs) => {
                console.log(`Batch ${batch + 1} completed.`);
            }
        }
    });

    progressBar.value = 100;
    trainingProgress.innerText = "Training complete!";
}

// 모델 로딩 후 훈련 시작 (예시)
loadModel().then(trainModel);
