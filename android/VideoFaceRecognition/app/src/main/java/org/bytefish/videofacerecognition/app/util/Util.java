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

package org.bytefish.videofacerecognition.app.util;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.util.Base64;
import android.view.OrientationEventListener;
import android.view.Surface;

import java.io.ByteArrayOutputStream;

/**
 * This class uses Utility functions written for the Camera module of Android.
 * These snippets have been taken from:
 *
 *      https://android.googlesource.com/platform/packages/apps/Camera/
 *
 *  Android code is released under terms of the Apache 2.0 license. You can obtain the copy in
 *  the assets folder coming with this project.
 *
 *  Copyright (C) 2011 The Android Open Source Project
 *
 */
public class Util {

    // Orientation hysteresis amount used in rounding, in degrees
    private static final int ORIENTATION_HYSTERESIS = 5;

    /**
     * Gets the current display rotation in angles.
     *
     * @param activity
     * @return
     */
    public static int getDisplayRotation(Activity activity) {
        int rotation = activity.getWindowManager().getDefaultDisplay()
                .getRotation();
        switch (rotation) {
            case Surface.ROTATION_0: return 0;
            case Surface.ROTATION_90: return 90;
            case Surface.ROTATION_180: return 180;
            case Surface.ROTATION_270: return 270;
        }
        return 0;
    }

    /**
     * Returns the Base64 representation of a Bitmap, which is compressed with the given format
     * and quality.
     *
     * @param bitmap
     * @param quality
     * @return
     */
    public static String getBase64(Bitmap bitmap, Bitmap.CompressFormat format, int quality) {
        ByteArrayOutputStream full_stream = new ByteArrayOutputStream();
        bitmap.compress(format, quality, full_stream);
        byte[] full_bytes = full_stream.toByteArray();
        return Base64.encodeToString(full_bytes, Base64.DEFAULT);
    }

    public static Bitmap rotateBitmap(Bitmap bitmap, int rotation) {
        Matrix matrix = new Matrix();
        matrix.postRotate(rotation);
        // create a new bitmap from the original using the matrix to transform the result
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    /**
     * Extracts a Rectangle given by a Camera.Face from a given Bitmap.
     *
     * @param bitmap Input Source Image
     * @param face Face Detection result
     * @return Cropped face
     */
    public static Bitmap extract(Bitmap bitmap, Camera.Face face, int rotation) {
        // The coordinates of the Camera.Face are given in a range of (-1000,1000),
        // so let's scale them to the Bitmap coordinate system:
        Matrix matrix = new Matrix();

        matrix.postScale(bitmap.getWidth() / 2000f, bitmap.getHeight() / 2000f);
        matrix.postTranslate(bitmap.getWidth() / 2f, bitmap.getHeight() / 2f);

        // Now translate the Camera.Face coordinates into the
        // Bitmap coordinate system:
        RectF scaledRect = new RectF(face.rect);
        matrix.mapRect(scaledRect);
        // And make a Rect again, we need it later. It's the source we want
        // to crop from:
        Rect srcRect = new Rect((int) scaledRect.left, (int) scaledRect.top, (int)scaledRect.right, (int) scaledRect.bottom );
        // This is the destination rectangle, we want it to have the width
        // and height of the scaled rect:
        int width = (int) scaledRect.width();
        int height = (int) scaledRect.height();
        Rect dstRect = new Rect(0, 0, width, height);
        // This is the output image, which is going to store the Camera.Face:
        Bitmap croppedImage = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        // And finally crop the image, which is a simple drawBitmap call on the
        // Canvas:
        Canvas canvas = new Canvas(croppedImage);
        canvas.drawBitmap(bitmap, srcRect, dstRect, null);

        return rotateBitmap(croppedImage, rotation);
    }

    /**
     * Converts a YUV image into a Jpeg Bitmap. This is because the Camera Preview returns
     * a YUV Image, from which we are going to build a JPEG image from. It's better to use
     * Androids converter, instead of rolling our own.
     *
     * @param data Image in YUV.
     * @param camera
     * @return
     */
    public static Bitmap convertYuvByteArrayToBitmap(byte[] data, Camera camera) {
        Bitmap result = null;
        if(camera != null) {
            Camera.Parameters parameters = camera.getParameters();
            Camera.Size size = parameters.getPreviewSize();
            result = convertYuvByteArrayToBitmap(data, parameters, size);
        }
        return result;
    }

    private static Bitmap convertYuvByteArrayToBitmap(byte[] data, Camera.Parameters cameraParameters, Camera.Size cameraSize) {
        YuvImage image = new YuvImage(data, cameraParameters.getPreviewFormat(), cameraSize.width, cameraSize.height, null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        image.compressToJpeg(new Rect(0, 0, cameraSize.width, cameraSize.height), 100, out);
        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    public static int getDisplayOrientation(int degrees, int cameraId) {
        // See android.hardware.Camera.setDisplayOrientation for
        // documentation.
        Camera.CameraInfo info = new Camera.CameraInfo();
        Camera.getCameraInfo(cameraId, info);
        int result;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (info.orientation + degrees) % 360;
            result = (360 - result) % 360;  // compensate the mirror
        } else {  // back-facing
            result = (info.orientation - degrees + 360) % 360;
        }
        return result;
    }

    public static void prepareMatrix(Matrix matrix, boolean mirror, int displayOrientation,
                                     int viewWidth, int viewHeight) {
        // Need mirror for front camera.
        matrix.setScale(mirror ? -1 : 1, 1);
        // This is the value for android.hardware.Camera.setDisplayOrientation.
        matrix.postRotate(displayOrientation);
        // Camera driver coordinates range from (-1000, -1000) to (1000, 1000).
        // UI coordinates range from (0, 0) to (width, height).
        matrix.postScale(viewWidth / 2000f, viewHeight / 2000f);
        matrix.postTranslate(viewWidth / 2f, viewHeight / 2f);
    }

    public static int roundOrientation(int orientation, int orientationHistory) {
        boolean changeOrientation = false;
        if (orientationHistory == OrientationEventListener.ORIENTATION_UNKNOWN) {
            changeOrientation = true;
        } else {
            int dist = Math.abs(orientation - orientationHistory);
            dist = Math.min( dist, 360 - dist );
            changeOrientation = ( dist >= 45 + ORIENTATION_HYSTERESIS );
        }
        if (changeOrientation) {
            return ((orientation + 45) / 90 * 90) % 360;
        }
        return orientationHistory;
    }
}
