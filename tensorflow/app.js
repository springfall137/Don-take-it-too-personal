let model;

async function loadModel() {
    model = await tf.loadLayersModel('model_js/model.json');
    document.getElementById('result').innerText = 'Model loaded, ready to predict!';
}

document.getElementById('image-upload').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = async function (event) {
            const imgElement = document.createElement('img');
            imgElement.src = event.target.result;
            imgElement.onload = async function () {
                const tensor = tf.browser.fromPixels(imgElement)
                    .resizeNearestNeighbor([150, 150])
                    .toFloat()
                    .expandDims();
                const predictions = await model.predict(tensor).data();
                const isDog = predictions[0] > 0.5;
                document.getElementById('result').innerText = isDog ? 'It\'s a Dog!' : 'It\'s a Cat!';
            }
        };
        reader.readAsDataURL(file);
    }
});

loadModel();
