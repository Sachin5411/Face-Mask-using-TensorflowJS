// import * as tf from '@tensorflow/tfjs';
// import * as tfd from '@tensorflow/tfjs-data';
// import * as blazeface from '@tensorflow-models/blazeface';

let video, videoWidth, videoHeight;
var web_url_for_model=window.location.href;   // to get url 

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

let canvas, canvasCtx;
async function setupCanvas() {
  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  canvasCtx = canvas.getContext('2d');
  canvasCtx.fillStyle = "rgba(255, 0, 0, 0.5)";
}

let faceDetectionModel;
async function loadFaceDetectionModel() {
  console.log("loading face detection model");
  await blazeface.load().then(m => {
    faceDetectionModel = m;
    console.log("face detection model loaded");
  });
}

let maskDetectionModel;
async function loadMaskDetectionModel() {
  console.log("loading mask detection model");

  console.log("using pre-trained model")
  await tf.loadLayersModel('model.json').then(m => {
    maskDetectionModel = m;
    console.log("Pre trained mask detection model loaded");
  });
}


const returnTensors = false;
const flipHorizontal = true;
const annotateBoxes = false;
const offset = tf.scalar(127.5);

const decisionThreshold = 0.9;

const loadingModel = document.getElementById('loading-model');

async function renderPrediction() {
  // Get image from webcame
  let img = tf.tidy(() => tf.browser.fromPixels(video));

  // Detect faces
  let faces = [];
  try {
    faces = await faceDetectionModel.estimateFaces(img, returnTensors, flipHorizontal, annotateBoxes);
  } catch (e) {
    console.error("estimateFaces:", e);
    return;
  }

  if (faces.length > 0) {
    // TODO: Loop through all predicted faces and detect if mask used or not.
    // RIght now, it only highlights the fisrt face into the live view. (See the break command below)
    for (let i = 0; i < faces.length; i++) {
      let predictions = [];

      let face = tf.tidy(() => img.resizeNearestNeighbor([224, 224])
        .toFloat().sub(offset).div(offset).expandDims(0));

      try {
        predictions = await maskDetectionModel.predict(face).data();
      } catch (e){
        console.error("maskDetection:", e);
        return;
      }

      face.dispose();

      const start = faces[i].topLeft;
      const end = faces[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];

      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

      let faceBoxStyle = "rgba(255, 0, 0, 0.25)";
      let label = "without mask";
      if (predictions.length > 0) {
        if (predictions[0] > decisionThreshold) {
          faceBoxStyle = "rgba(0, 255, 0, 0.25)";
          label = `Mask Found: ${Math.floor(predictions[0] * 1000) / 10}%`;
        } else {
          label = `No Mask Found : ${Math.floor(predictions[1] * 1000) / 10}%`;
        }

        // Render label and its box
        canvasCtx.fillStyle = "rgba(255, 111, 0, 0.85)";
        canvasCtx.fillRect(start[0], start[1] - 23, size[0], 23);
        canvasCtx.font = "18px Raleway";
        canvasCtx.fillStyle = "rgba(255, 255, 255, 1)";
        canvasCtx.fillText(label, end[0] + 5, start[1] - 5);
      }

      canvasCtx.fillStyle = faceBoxStyle;
      canvasCtx.fillRect(start[0], start[1], size[0], size[1]);

      // TODO: Loop through all detected faces instead of the first one.
      break;  
    }
  }

  img.dispose();
  requestAnimationFrame(renderPrediction);

  // if (loadingModel.innerHTML !== "") {
  //   loadingModel.innerHTML = "";
  // }
}

async function user_model(){
  const modelfile = document.getElementById('modelUpload').files[0];

if (modelfile) {
        console.log("Using User Uploaded Model")
        var link=web_url_for_model+modelfile.name
        console.log(link)
       await tf.loadLayersModel(link).then(m => {
    maskDetectionModel = m;
    console.log("User's mask detection model loaded");
  }); 
    }

  else{
    alert("Seems like you didn't upload a model")
  }
}

//If user uploads a model
async function main_user() {
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  setupCanvas();

  await loadFaceDetectionModel();
  await user_model();

  renderPrediction();
}

//for pretrained model
async function main() {
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  setupCanvas();

  await loadFaceDetectionModel();
  await loadMaskDetectionModel();

  renderPrediction();
}

// function clearmodel(){
  
//   button_value=document.getElementById('change_upload').value
//   if(button_value=="Uploaded"){
//     document.getElementById("modelUpload").value = "";
//   document.getElementById('change_upload').value="Upload Model"
  
// }
// var buto = document.querySelector('.Uploadyourmodel');
//   buto.style.display = 'none';

// }

function upload_model(){
  const modelfile2 = document.getElementById('modelUpload').files[0];
  if (modelfile2){
  // document.getElementById('change_upload').value="Uploaded"
  loadingModel.innerHTML=""
  loadingModel.innerHTML="Using User's Model"
  main_user ();
}
  else{
    alert("Please Select a model First :)")
    console.log("You didn't upload the model")
  }
}

const upl_btn=document.getElementById('user-upload')
const realfile = document.getElementById('modelUpload');


// async function uploadedmodel() {
upl_btn.addEventListener('click',function(){
realfile.click();

});


realfile.addEventListener('change',function(){
console.log("model Selected")
upload_model();

})
// }

function pretrainedmodel(){

  loadingModel.innerHTML=""
  loadingModel.innerHTML="Using Pre-trained Model";
  document.getElementById('live-view').style.cssText = 'background-size: 0 0;position: relative;height: 42%;width: 55%;margin-left : 22%;';
  main();
}

window.addEventListener('load', bindEvents3);

var popOut = document.querySelector('.bg-popOutPg');
var hidePopOut = document.querySelector('.lbtbBtn');

function bindEvents3() {
    hidePopOut.addEventListener('click', hidePop);
    // popOut.addEventListener('click',hidePop);
}

function hidePop() {
    popOut.style.display = 'none';
}
// main();
