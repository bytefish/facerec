/**
 * Copyright (c) 2014, Philipp Wagner <bytefish(at)gmx(dot)de>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of
 * conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of
 * conditions and the following disclaimer in the documentation and/or other materials provided
 * with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package org.bytefish.videofacerecognition.app;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import android.app.ActionBar;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.hardware.Camera.Face;
import android.hardware.Camera.FaceDetectionListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.v7.app.ActionBarActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.OrientationEventListener;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.ViewGroup.LayoutParams;
import android.widget.Toast;

import org.bytefish.videofacerecognition.api.client.FaceRecServiceClient;
import org.bytefish.videofacerecognition.app.common.Constants;
import org.bytefish.videofacerecognition.app.task.AsyncTaskResult;
import org.bytefish.videofacerecognition.app.webservice.FaceRecognitionCallback;
import org.bytefish.videofacerecognition.app.webservice.FaceRecognitionRequest;
import org.bytefish.videofacerecognition.app.webservice.FaceRecognitionRequestYuv;
import org.bytefish.videofacerecognition.app.webservice.FaceRecognitionResult;
import org.bytefish.videofacerecognition.app.webservice.FaceRecognitionTask;
import org.bytefish.videofacerecognition.app.util.Util;
import org.bytefish.videofacerecognition.app.view.FaceOverlayView;

import javax.xml.datatype.Duration;


public class CameraActivity extends Activity
        implements SurfaceHolder.Callback, Camera.PreviewCallback {

    public static final String TAG = CameraActivity.class.getSimpleName();

    private Camera mCamera;

    // We need the phone orientation to correctly draw the Overlay
    private int mOrientation;
    private int mOrientationCompensation;
    private OrientationEventListener mOrientationEventListener;

    // Also
    private int mDisplayRotation;
    private int mDisplayOrientation;

    // The surface view for the camera data
    private SurfaceView mView;

    // Draw rectangles and other fancy stuff:
    private FaceOverlayView mFaceView;

    // Holds the current frame, so we can react on a click event:
    private final Lock lock = new ReentrantLock();
    private byte[] mPreviewFrameBuffer;

    // FaceRecognition Service:
    FaceRecServiceClient mFaceRecServiceClient;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Initialize the app:
        initActionBar();
        initSettings();
        // SurfaceView holding the Camera Preview:
        mView = new SurfaceView(this);
        // Set the Camera Preview as Content:
        setContentView(mView);
        // Now create the OverlayView:
        mFaceView = new FaceOverlayView(this);
        addContentView(mFaceView, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));
        // Create and Start the OrientationListener:
        mOrientationEventListener = new SimpleOrientationEventListener(this);
        mOrientationEventListener.enable();
    }

    private void initSettings() {
        Log.i(TAG, "Updating settings.");

        SharedPreferences sharedPrefs = PreferenceManager.getDefaultSharedPreferences(this);

        String server_address = sharedPrefs.getString("key_server_address", "http://192.168.178.21:5000");
        boolean server_enable_https = sharedPrefs.getBoolean("key_enable_https", false);

        // Create a new FaceRecService Client:
        mFaceRecServiceClient = new FaceRecServiceClient(server_address, null, null);
    }

    private void initActionBar() {
        ActionBar actionBar = getActionBar();
        if (actionBar != null) {
            actionBar.setHomeButtonEnabled(false); // disable the button
            actionBar.setDisplayHomeAsUpEnabled(false); // remove the left caret
            actionBar.setDisplayShowHomeEnabled(false); // remove the icon
            actionBar.setDisplayShowTitleEnabled(false);
        }
    }


    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle presses on the action bar items
        switch (item.getItemId()) {
            case R.id.action_settings:
                Log.i(TAG, "Settings selected");
                Intent intent = new Intent(CameraActivity.this, SettingsActivity.class);
                startActivityForResult(intent, Constants.REQUEST_CODE_SETTINGS);
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == Constants.REQUEST_CODE_SETTINGS) {
            initSettings();
        }

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu items for use in the action bar
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.main, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public void onPreviewFrame(byte[] bytes, Camera camera) {
        try {
            lock.lock();
            mPreviewFrameBuffer = bytes;
        } finally {
            lock.unlock();
        }
    }

    /**
     * We need to react on OrientationEvents to rotate the screen and
     * update the views.
     */
    private class SimpleOrientationEventListener extends OrientationEventListener {

        public SimpleOrientationEventListener(Context context) {
            super(context, SensorManager.SENSOR_DELAY_NORMAL);
        }

        @Override
        public void onOrientationChanged(int orientation) {
            // We keep the last known orientation. So if the user first orient
            // the camera then point the camera to floor or sky, we still have
            // the correct orientation.
            if (orientation == ORIENTATION_UNKNOWN) return;
            mOrientation = Util.roundOrientation(orientation, mOrientation);
            // When the screen is unlocked, display rotation may change. Always
            // calculate the up-to-date orientationCompensation.
            int orientationCompensation = mOrientation
                    + Util.getDisplayRotation(CameraActivity.this);
            if (mOrientationCompensation != orientationCompensation) {
                mOrientationCompensation = orientationCompensation;
                // Update the Overlay with the current orientation:
                if(mFaceView != null) {
                    mFaceView.setOrientation(mOrientationCompensation);
                }
            }
        }
    }

    /**
     * Store the face data, so we can start the AsyncTask for the face recognition
     * process instantly.
     */
    private FaceDetectionListener faceDetectionListener = new FaceDetectionListener() {
        @Override
        public void onFaceDetection(Face[] faces, Camera camera) {
            Log.d("onFaceDetection", "Number of Faces:" + faces.length);
            // Update the view now!
            mFaceView.setFaces(faces);
        }
    };


    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
        SurfaceHolder holder = mView.getHolder();
        holder.addCallback(this);
    }

    @Override
    protected void onPause() {
        mOrientationEventListener.disable();
        if(mCamera != null) {
            mCamera.stopPreview();
        }
        super.onPause();
    }

    @Override
    protected void onResume() {
        mOrientationEventListener.enable();
        super.onResume();
    }

    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {
        mCamera = Camera.open();
        mCamera.setFaceDetectionListener(faceDetectionListener);
        mCamera.startFaceDetection();
        try {
            mCamera.setPreviewDisplay(surfaceHolder);
            mCamera.setPreviewCallback(this);
        } catch (Exception e) {
            Log.e(TAG, "Could not preview the image.", e);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        int x = (int)event.getX();
        int y = (int)event.getY();
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                break;
            case MotionEvent.ACTION_MOVE:
                break;
            case MotionEvent.ACTION_UP:
            {
                Face face = mFaceView.touchIntersectsFace(x,y);
                if(face != null) {
                    //Toast.makeText(getApplicationContext(), "(" + x + "," + y +")", Toast.LENGTH_LONG).show();
                    try {
                        lock.lock();
                        // Process the buffered frame! This is safe, because we have locked the access
                        // to the resource we are going to work on. This task should be a background
                        // task, in case it takes too long.
                        UUID requestIdentifier = UUID.randomUUID();
                        // Create the Request Object:
                        FaceRecognitionRequest request = new FaceRecognitionRequestYuv(requestIdentifier, mPreviewFrameBuffer.clone(), mCamera, face, mDisplayOrientation);
                        // Execute a FaceRecognitionRequest:
                        new FaceRecognitionTask(faceRecognitionCallback, mFaceRecServiceClient).execute(request);
                    } finally {
                        lock.unlock();
                    }
                }
            }
            break;
        }
        return false;
    }


    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i2, int i3) {
        // We have no surface, return immediately:
        if (surfaceHolder.getSurface() == null) {
            return;
        }
        // Try to stop the current preview:
        try {
            mCamera.stopPreview();
        } catch (Exception e) {
            // Ignore...
        }
        // Get the supported preview sizes:
        Camera.Parameters parameters = mCamera.getParameters();
        List<Camera.Size> previewSizes = parameters.getSupportedPreviewSizes();
        Camera.Size previewSize = previewSizes.get(0);
        // And set them:
        parameters.setPreviewSize(previewSize.width, previewSize.height);
        mCamera.setParameters(parameters);
        // Now set the display orientation for the camera. Can we do this differently?
        mDisplayRotation = Util.getDisplayRotation(CameraActivity.this);
        mDisplayOrientation = Util.getDisplayOrientation(mDisplayRotation, 0);
        mCamera.setDisplayOrientation(mDisplayOrientation);

        if (mFaceView != null) {
            mFaceView.setDisplayOrientation(mDisplayOrientation);
        }

        // Finally start the camera preview again:
        mCamera.setPreviewCallback(this);
        mCamera.startPreview();
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
        mCamera.setPreviewCallback(null);
        mCamera.setFaceDetectionListener(null);
        mCamera.setErrorCallback(null);
        mCamera.release();
        mCamera = null;
    }

    FaceRecognitionCallback faceRecognitionCallback = new FaceRecognitionCallback() {
        @Override
        public void OnCompleted(FaceRecognitionResult result) {
            Log.i(TAG, "Face Recognition completed (uuid=" + result.getRequestIdentifier() + ", result=" + result.getResult());
            Toast.makeText(getApplicationContext(), "Name: " + result.getResult(), Toast.LENGTH_LONG).show();
        }

        @Override
        public void OnFailed(Exception exception) {
            Log.e(TAG, "Face recognition has failed!", exception);
            Toast.makeText(getApplicationContext(), "Face Recognition failed.", Toast.LENGTH_LONG).show();
        }

        @Override
        public void OnCanceled() {
            Log.e(TAG, "Face recognition was canceled!");
            Toast.makeText(getApplicationContext(), "Face Recognition failed.", Toast.LENGTH_LONG).show();
        }
    };
}