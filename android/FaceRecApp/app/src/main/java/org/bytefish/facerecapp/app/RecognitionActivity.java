package org.bytefish.facerecapp.app;

import android.app.ActionBar;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.drawable.BitmapDrawable;
import android.media.FaceDetector;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.widget.ImageView;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

public class RecognitionActivity extends Activity {

    private static final String TAG = RecognitionActivity.class.getName();

    // We want to detect atmost 5 faces:
    private static final int maxNumberOfFaces = 5;

    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recognition);

        // Get the image view:

        // Add action bar:
        ActionBar actionBar = getActionBar();
        actionBar.setDisplayHomeAsUpEnabled(true);

        // Get fileName from Intent:
        Intent myIntent = getIntent();
        String fileName = myIntent.getStringExtra(Constants.RECOGNIZE_ACTIVITY_IMAGE_PATH); // will return "FirstKeyValue"

        // Get the ImageView:
        imageView = (ImageView) findViewById(R.id.recognition_results);

        // And now detect the faces!
        FaceDetectionTask faceDetectionTask = new FaceDetectionTask(imageView);
        faceDetectionTask.execute(new String[] { fileName });
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                // go to previous screen when app icon in action bar is clicked
                Intent intent = new Intent(this, MainActivity.class);
                intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                startActivity(intent);
                return true;
        }
        return super.onOptionsItemSelected(item);
    }

    /**
     * Holds the FaceDetection result. For now it is simply the bounding box of the Face and the
     * predicted name of the face.
     */
    private class FaceRecognitionResult {

        public FaceDetector.Face face;
        public String name;

        public FaceRecognitionResult(FaceDetector.Face face, String name) {
            this.face = face;
            this.name = name;
        }
    }

    /**
     *
     * Performs the FaceDetection part in Background, because it might be a long running Task.
     * This implementation might be problematic, because orientation changes (which are not
     * unlikely) might kill this Task.
     *
     * TODO Publish Progress
     * TODO Show Image before performing face detection/face recognition
     *
     */
    private class FaceDetectionTask extends AsyncTask<String, String, List<FaceRecognitionResult>> {

        private final WeakReference<ImageView> imageViewReference;
        private Bitmap image;

        public FaceDetectionTask(ImageView imageView) {
            imageViewReference = new WeakReference<ImageView>(imageView);
        }

        @Override
        protected List<FaceRecognitionResult> doInBackground(String... fileNames) {
            String fileName = fileNames[0];
            image = ImageHelper.readBitmapFromFile(fileName);
            // Detect faces:
            FaceDetector.Face[] faces = FaceDetectionHelper.detectFaces(image);
            // Recognize each face:
            List<FaceRecognitionResult> faceRecognitionResults = new ArrayList<FaceRecognitionResult>();
            for(int pos = 0; pos < faces.length; pos++) {
                try {
                    FaceDetector.Face face = faces[pos];
                    String name = FaceRecognitionHelper.recognizeFace(image, faces[pos]);
                    // Add to the result list:
                    faceRecognitionResults.add(new FaceRecognitionResult(face, name));
                } catch(Exception e) {
                    Log.e(TAG, "Face Recognition failed. No result added.", e);
                }
            }
            return faceRecognitionResults;
        }

        private Paint getBoundingBoxPaint() {
            Paint boxPaint = new Paint();

            boxPaint.setColor(Color.GREEN);
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(3);

            return boxPaint;
        }

        private Paint getTextPaint() {
            Paint textPaint = new Paint();

            textPaint.setColor(Color.GREEN);
            textPaint.setStyle(Paint.Style.FILL);
            textPaint.setStrokeWidth(1);
            textPaint.setTextSize(20);

            return textPaint;
        }

        @Override
        protected void onPostExecute(List<FaceRecognitionResult> result) {
            ImageView imageView = imageViewReference.get();
            if(imageView != null && image != null) {
                // Get colors and styles:
                Paint boxPaint = getBoundingBoxPaint();
                Paint textPaint = getTextPaint();
                // We want to draw on it, so we need a mutable image:
                Bitmap mutableImage = image.copy(image.getConfig(), true);
                // Now put it inside a Canvas and draw on it:
                Canvas canvas = new Canvas(mutableImage);
                // Append information to each face:
                for (int i = 0; i < result.size(); i++) {
                    FaceDetector.Face face = result.get(i).face;

                    // Get Face bounding box:
                    PointF myMidPoint = new PointF();
                    face.getMidPoint(myMidPoint);
                    float myEyesDistance = face.eyesDistance();

                    int x0 = (int) (myMidPoint.x - myEyesDistance * 2);
                    int y0 = (int) (myMidPoint.y - myEyesDistance * 2);
                    int x1 = (int) (myMidPoint.x + myEyesDistance * 2);
                    int y1 = (int) (myMidPoint.y + myEyesDistance * 2);

                    // Clip to image boundaries
                    x0 = Math.max(x0, 0);
                    y0 = Math.max(y0, 0);
                    x1 = Math.min(x1, mutableImage.getWidth());
                    y1 = Math.min(y1, mutableImage.getHeight());


                    // Draw bounding rectangle into canvas:
                    canvas.drawRect(x0, y0, x1, y1, boxPaint);

                    // Add name next to rectangle:
                    String name = result.get(i).name;
                    canvas.drawText(name, x1, y0, textPaint);

                }
                imageView.setImageBitmap(mutableImage);
            }
        }
    }
}
