package org.bytefish.facerecapp.app;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.v7.app.ActionBarActivity;
import android.view.Menu;
import android.view.MenuItem;

/**
 * Captures a new image from camera
 */
public class CaptureActivity extends Activity {

    private static final int CAPTURE_IMAGE_REQUEST_CODE = 100;
    Uri fileUri = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Create Intent to take a picture and return control to the calling application:
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Create a new and unique filename to save the image to:
        fileUri = ImageHelper.getOutputImageFileUri();
        // We want to store the image there:
        intent.putExtra(MediaStore.EXTRA_OUTPUT, fileUri);
        // And now start the image capture Intent!
        startActivityForResult(intent, CAPTURE_IMAGE_REQUEST_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == CAPTURE_IMAGE_REQUEST_CODE) {
            if (resultCode == RESULT_OK) {
                // We got the image, start recognition process!
                callRecognitionActivity();
            } else {
                Intent intent = new Intent(this, MainActivity.class);
                intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                startActivity(intent);
            }
        }
    }

    private void callRecognitionActivity() {
        Intent recognitionIntent = new Intent(CaptureActivity.this, RecognitionActivity.class);
        recognitionIntent.putExtra(Constants.RECOGNIZE_ACTIVITY_IMAGE_PATH, fileUri.toString());
        startActivity(recognitionIntent);
    }
}
