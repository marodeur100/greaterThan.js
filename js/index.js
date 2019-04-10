const video = document.getElementById("myvideo");
// const handimg = document.getElementById("handimage");
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const canvas2 = document.getElementById("canvas2");
const context2 = canvas2.getContext("2d");
let trackButton = document.getElementById("trackbutton");
// let nextImageButton = document.getElementById("nextimagebutton");
let updateNote = document.getElementById("updatenote");
var vc = new cv.VideoCapture(video);

let imgindex = 1
let isVideo = false;
let model = null;


const modelParams = {
    flipHorizontal: true,   // flip e.g for video  
    maxNumBoxes: 20,        // maximum number of boxes to detect
    iouThreshold: 0.5,      // ioU threshold for non-max suppression
    scoreThreshold: 0.6,    // confidence threshold for predictions.
}

function startVideo() {
    handTrack.startVideo(video).then(function (status) {
        console.log("video started", status);
        if (status) {
            updateNote.innerText = "Video camera. Now tracking"
            isVideo = true
            runDetection()
        } else {
            updateNote.innerText = "Please enable camera"
        }
    });
}

function toggleVideo() {
    if (!isVideo) {
        updateNote.innerText = "Starting camera"
        startVideo();
    } else {
        updateNote.innerText = "Stopping camera"
        handTrack.stopVideo(video)
        isVideo = false;
        updateNote.innerText = "Camera stopped"
    }
}


trackButton.addEventListener("click", function(){
    toggleVideo();
});


function runDetection() {


    model.detect(video).then(predictions => {
        greaterThanFound = false;

        // show video
        context2.drawImage(video,0,0,video.width,video.height);
        var img = cv.matFromImageData(context2.getImageData(0,0,video.width,video.height));
        cv.flip(img, img, 1);
        cv.imshow('canvas2', img);

        // find contour
        for(i=0;i<predictions.length;i++ ) {
            x1 = predictions[i].bbox[0];
            y1 = predictions[i].bbox[1];
            x2 = predictions[i].bbox[2];
            y2 = predictions[i].bbox[3];

            //let dsize = new cv.Size(video.width, video.height);
            //cv.resize(img, img, dsize, 0, 0, cv.INTER_AREA);

            // get sub image from canvas
            imgData = context2.getImageData(x1, y1, x2, y2);
            // find biggest contour
            contour = this.detectFingers(cv.matFromImageData(imgData), x1, y1, x2, y2);
            // render greater than
            greaterThanFound = drawGreaterThan(contour, x1, y1, x2, y2, img)

            // console.log("Predictions: ", predictions);
            if (!greaterThanFound) {
                model.renderPredictions(predictions, canvas2, context2, video);
            }
        }

        if (isVideo) {
            requestAnimationFrame(runDetection);
        }
    });
}


// Load the model.
handTrack.load(modelParams).then(lmodel => {
    // detect objects in the image.
    video.style.display = "none";
    model = lmodel
    updateNote.innerText = "Loaded Model!"
    //runDetectionImage(handimg)
    trackButton.disabled = false
});


function detectFingers(src, x1, y1, x2, y2){
    //backproj = backProject(x1, y1, x2, y2);
    //cv.imshow('canvas2', backproj);
    let anchor = new cv.Point(-1, -1);
    cv.cvtColor(src, src, cv.COLOR_BGR2HLS, 0);
    let low = new cv.Mat(src.rows, src.cols, src.type(), [0, 0, 0, 0]);
    let high = new cv.Mat(src.rows, src.cols, src.type(), [140, 140, 140, 255]);
    cv.inRange(src, low, high, src);

    cv.blur(src, src, new cv.Size(10, 10), anchor, cv.BORDER_DEFAULT);
    cv.threshold(src, src, 200, 255, cv.THRESH_BINARY);
    //cv.bitwise_not(src, src);

    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();

    cv.findContours(src, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    if (contours) {
        candidate = findCandidate(contours, 0, 0);
    }
    // cv.imshow('canvas2', src);
    return candidate;
};
  
function findCandidate (contours, minSize, epsilon){
    contour = this.findMaxArea(contours, minSize);
    if (contour){
        //cv.approxPolyDP(contour, contour, 3, true);
        return contour;
    }
    return null;
  };
  
function findMaxArea (contours, minSize){
    var len = contours.size(), i = 0,
        maxArea = -1, area = 0, contour;
    for (; i < len; ++ i){
        area = cv.contourArea(contours.get(i));
        if (area > maxArea) {
          maxArea = area;
          contour = contours.get(i);
        }
    }
    return contour;
  };


function drawGreaterThan(contour, x1, y1, x2, y2, img) {
    var greaterThanFound = false;
    var img_orig = img.clone();
    let lineColor = new cv.Scalar(0, 255, 0);
    if (contour) {
        let tmp = new cv.Mat();
        let defect = new cv.Mat();
        cv.convexHull(contour, tmp, false, false);
        cv.convexityDefects(contour, tmp, defect);
        // console.log("Defects: ", defect);
        var end_c= new cv.Point(0,0), start_c= new cv.Point(0,0), far_c= new cv.Point(0,0);
        let cnt = 0;
        if (defect) {
            for (let i = 0; i < defect.rows; ++i) {
                let start = new cv.Point(x1 + contour.data32S[defect.data32S[i * 4] * 2],
                                        y1 + contour.data32S[defect.data32S[i * 4] * 2 + 1]);
                let end = new cv.Point(x1 + contour.data32S[defect.data32S[i * 4 + 1] * 2],
                                    y1 + contour.data32S[defect.data32S[i * 4 + 1] * 2 + 1]);
                let far = new cv.Point(x1 + contour.data32S[defect.data32S[i * 4 + 2] * 2],
                                    y1 + contour.data32S[defect.data32S[i * 4 + 2] * 2 + 1]);
                let a = Math.sqrt(((end.x - start.x) ** 2) + ((end.y - start.y) ** 2));
                let b = Math.sqrt(((far.x - start.x) ** 2) + ((far.y - start.y) ** 2));
                let c = Math.sqrt(((end.x - far.x) ** 2) + ((end.y - far.y) ** 2));
                angle = Math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c));
                cv.line(img, start, far, lineColor, 4, cv.LINE_AA, 0);
                cv.line(img, far, end, lineColor, 4, cv.LINE_AA, 0);
                // angle less than 90 degree, treat as fingers
                if (angle <= (Math.PI / 2)) { 
                    cnt++;
                    start_c = start;
                    end_c = end;
                    far_c= far;
                }
            }

            
            if (cnt==1) {
                if ((far_c.x>start_c.x) && (far_c.y>start_c.y) && (far_c.y<end_c.y) ) {
                    img = img_orig.clone();
                    cv.line(img, new cv.Point(start_c.x, end_c.y), new cv.Point(far_c.x, start_c.y+((end_c.y-start_c.y)/2)), lineColor, 4, cv.LINE_AA, 0);
                    cv.line(img, start_c, new cv.Point(far_c.x, start_c.y+((end_c.y-start_c.y)/2)), lineColor, 4, cv.LINE_AA, 0);
                    cv.putText(img, 'IES ASG rocks!', new cv.Point(start_c.x, (start_c.y - 10)), cv.FONT_HERSHEY_PLAIN, 1, lineColor, 2);
                    greaterThanFound = true;
                }
            }

        }
    }
    cv.imshow('canvas2', img);
    img.delete();
    img_orig.delete();
    return greaterThanFound;
}


function draw (contour, image){
    if (contour){
        let tmp = new cv.Mat();
        let hull = new cv.MatVector();
        cv.convexHull(contour, tmp, false, true);
        // console.log("Hull: ", tmp);
        hull.push_back(tmp);
        let defect = new cv.Mat();
        cv.convexHull(contour, tmp, false, false);
        cv.convexityDefects(contour, tmp, defect);
        // console.log("Defects: ", defect);
        this.drawHullAndDefects(hull, defect, image, contour);
    }
    //return image;
  };
  
function drawHullAndDefects(hull, defect, src, cnt){
    // draw contours with random Scalar
    let dst = src;
    let hierarchy = new cv.Mat();
    let colorHull = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                                Math.round(Math.random() * 255));
    cv.drawContours(dst, hull, 0, colorHull, 1, 8, hierarchy, 0);
    let lineColor = new cv.Scalar(255, 0, 0);
    let circleColor = new cv.Scalar(255, 255, 255);
    for (let i = 0; i < defect.rows; ++i) {
        let start = new cv.Point(cnt.data32S[defect.data32S[i * 4] * 2],
                                 cnt.data32S[defect.data32S[i * 4] * 2 + 1]);
        let end = new cv.Point(cnt.data32S[defect.data32S[i * 4 + 1] * 2],
                               cnt.data32S[defect.data32S[i * 4 + 1] * 2 + 1]);
        let far = new cv.Point(cnt.data32S[defect.data32S[i * 4 + 2] * 2],
                               cnt.data32S[defect.data32S[i * 4 + 2] * 2 + 1]);
        cv.line(dst, end, far, lineColor, 2, cv.LINE_AA, 0);
        cv.line(dst, start, far, lineColor, 2, cv.LINE_AA, 0);
        cv.circle(dst, far, 3, circleColor, -1);
    }
    // cv.imshow('canvas2', dst);
};
  

function backProject(x1, y1, x2, y2)  {
    
    let src =  cv.matFromImageData(context.getImageData(x1 + ((x2-x1)/2*5), y1 + ((y2-y1)/2*5), x2- ((x2-x1)/2*5), y2 - ((y2-y1)/2*5)));
    let dst =  cv.matFromImageData(context.getImageData(x1, y1, x2, y2));

    cv.cvtColor(src, src, cv.COLOR_RGB2HSV, 0);
    cv.cvtColor(dst, dst, cv.COLOR_RGB2HSV, 0);
    let srcVec = new cv.MatVector();
    let dstVec = new cv.MatVector();
    srcVec.push_back(src); dstVec.push_back(dst);
    let backproj = new cv.Mat();
    let none = new cv.Mat();
    let mask = new cv.Mat();
    let hist = new cv.Mat();
    let channels = [0];
    let histSize = [50];
    let ranges = [0, 180];
    let accumulate = false;
    let anchor = new cv.Point(-1, -1);
    cv.calcHist(srcVec, channels, mask, hist, histSize, ranges, accumulate);
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX, -1, none);
    cv.calcBackProject(dstVec, channels, hist, backproj, ranges, 1);
    let M = new cv.Mat();
    let ksize = new cv.Size(31, 31);
    // M = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize);
    // cv.filter2D(backproj, backproj, cv.CV_8U, M, anchor, 0, cv.BORDER_DEFAULT);
    //cv.imshow('canvas2', backproj);
    //src.delete(); dst.delete(); srcVec.delete(); dstVec.delete();
    //backproj.delete(); mask.delete(); hist.delete(); none.delete();
    return backproj;
};
