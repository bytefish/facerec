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

package org.bytefish.facerecapp.app;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.media.ExifInterface;
import android.media.FaceDetector;
import android.net.Uri;
import android.os.Environment;
import android.util.Base64;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.UUID;

/**
 * Some useful functions for working with images in Android.
 */
public class ImageHelper {

    public static final String TAG = ImageHelper.class.getName();
    public static final String IM_DIR = "facerec";
    public static final String IM_PREFIX = "RECOGNIZE_";

    /**
     * @param originalBitmap Image to crop the face image from.
     * @param face           Landmarks found by Androids FaceDetector.
     * @return
     */
    public static Bitmap cropFace(Bitmap originalBitmap, FaceDetector.Face face) {
        // Get face midpoint:
        PointF faceMidPoint = new PointF();
        face.getMidPoint(faceMidPoint);
        float faceEyesDistance = face.eyesDistance();

        // Get rectangle:
        int x0 = (int) (faceMidPoint.x - faceEyesDistance * 2);
        int y0 = (int) (faceMidPoint.y - faceEyesDistance * 2);
        int x1 = (int) (faceMidPoint.x + faceEyesDistance * 2);
        int y1 = (int) (faceMidPoint.y + faceEyesDistance * 2);

        // Clip to image boundaries
        x0 = Math.max(x0, 0);
        y0 = Math.max(y0, 0);
        x1 = Math.min(x1, originalBitmap.getWidth());
        y1 = Math.min(y1, originalBitmap.getHeight());

        return Bitmap.createBitmap(originalBitmap, x0, y0, x1, y1);
    }

    /**
     * Returns a unique filename.
     *
     * @param fileExtension extension to append to the filename
     * @return filename
     */
    public static String getUniqueFileName(String fileExtension) {
        String uuid = UUID.randomUUID().toString();
        return (uuid + ".jpg");
    }

    /**
     * @param originalBitmap
     * @param x0
     * @param y0
     * @param x1
     * @param y1
     * @return
     */
    public static Bitmap cropImage(Bitmap originalBitmap, int x0, int y0, int x1, int y1) {
        return Bitmap.createBitmap(originalBitmap, x0, y0, x1, y1);
    }

    /**
     * Rotates a Bitmap for a given angle and rotation center.
     *
     * @param originalBitmap
     * @param cx
     * @param cy
     * @param angle
     * @return
     */
    public static Bitmap rotateBitmap(Bitmap originalBitmap, int cx, int cy, int angle) {
        Matrix rotMatrix = new Matrix();
        rotMatrix.postRotate(angle, cx, cy);
        return Bitmap.createBitmap(originalBitmap, 0, 0, originalBitmap.getWidth(), originalBitmap.getHeight(), rotMatrix, true);
    }

    /**
     * Creates a new Uri for the image to capture.
     *
     * @return filename for the captured image
     */
    public static Uri getOutputImageFileUri() {
        return Uri.fromFile(getOutputImageFile());
    }

    /**
     * Generates a new filename for the captured image.
     *
     * @return Filename
     */
    public static File getOutputImageFile() {
        File im_dir = getImageDirectory();
        // Create a new filename baed on the current timestamp:
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        // Create the file:
        File mediaFile = new File(im_dir.getPath() + File.separator + IM_PREFIX + timeStamp + ".jpg");

        return mediaFile;
    }

    /**
     * Reads an image into a Bitmap.
     *
     * @param fileName Filename of the given image
     * @return image in Bitmap representation
     */
    public static Bitmap readBitmapFromFile(String fileName) {
        BitmapFactory.Options bitmapFatoryOptions = new BitmapFactory.Options();
        bitmapFatoryOptions.inPreferredConfig = Bitmap.Config.RGB_565;
        return BitmapFactory.decodeFile(fileName, bitmapFatoryOptions);
    }

    /**
     * Saves a bitmap to the external storage. Please note, that you need the following
     * permission in your AndroidManifest.xml:
     * <p/>
     * <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
     *
     * @param bitmap
     * @param fileName
     * @return
     */
    public static boolean saveBitmapAsJpegToExternalStorage(Bitmap bitmap, String fileName) {
        File im_dir = getImageDirectory();
        File im_file = new File(im_dir, fileName);
        return saveBitmapAsJpeg(bitmap, im_file);
    }

    private static File getImageDirectory() {
        File root = Environment.getExternalStorageDirectory();
        File im_dir = new File(root, IM_DIR);
        if (!im_dir.exists()) {
            im_dir.mkdir();
        }
        return im_dir;
    }

    /**
     * Stores a Bitmap as a JPEG (with 90% Quality). This function should belong to
     * a class, which is configurable in terms of compression method, quality and so
     * on.
     *
     * @param bitmap Bitmap to store
     * @param file   File to store the image to
     * @return Returns the File
     */
    public static boolean saveBitmapAsJpeg(Bitmap bitmap, File file) {
        FileOutputStream out = null;
        try {
            // Overwrite the existing bitmap?
            out = new FileOutputStream(file, false);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
            out.flush();
        } catch (IOException e) {
            Log.e(TAG, "Could not save the image!", e);
            return false;
        } finally {
            // Be sure to close the IO handle:
            if (out != null) {
                try {
                    out.close();
                } catch (IOException e) {
                    // We can safely ignore this case!
                }
            }
        }
        return true;
    }

    /**
     * It's possible, that an image eats up so much memory, that the limits are exceeded and thus
     * our app is crashing. We want to prevent this by scaling the image down to a maximum size.
     * <p/>
     * The approach is a simple one, taken from these two links:
     * <p/>
     * http://developer.android.com/training/displaying-bitmaps/load-bitmap.html#load-bitmap
     * http://stackoverflow.com/questions/17839388/creating-a-scaled-bitmap-with-createscaledbitmap-in-android
     * <p/>
     * The trick is to not load the entire file in memory (using inJustDecodeBounds), then caclulate
     * the sample size for the decoder (plus one, as we don't want a smaller image than we have requested)
     * and finally scale it down to the maximum of the requestedWidth/requestedHeight (as we want to
     * respect the ratio of the image).
     *
     * @param fileName  File to read
     * @param reqWidth  Maximum width allowed
     * @param reqHeight Maximum height allowed
     * @return Scaled version without
     */
    public static Bitmap loadResizedBitmap(String fileName, int reqWidth, int reqHeight) {
        // First decode with inJustDecodeBounds=true to check dimensions
        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(fileName, options);
        // Calculate the maximum inSampleSize larger than the request width/height:
        options.inSampleSize = calculateInSampleSize(options, reqWidth, reqHeight) + 1;
        // Decode bitmap with inSampleSize set
        options.inJustDecodeBounds = false;
        options.inPreferredConfig = Bitmap.Config.RGB_565;
        Bitmap scaledBitmap = BitmapFactory.decodeFile(fileName, options);
        if (scaledBitmap.getHeight() > reqHeight || scaledBitmap.getWidth() > reqWidth) {
            return createScaledBitmap(scaledBitmap, reqWidth, reqHeight);
        }
        return scaledBitmap;
    }

    private static Bitmap createScaledBitmap(Bitmap bitmap, int reqWidth, int reqHeight) {
        // Calculate the maximum size:
        int maxSize = Math.max(reqWidth, reqHeight);
        // Now scale the width/height accordingly:
        int outWidth;
        int outHeight;
        int inWidth = bitmap.getWidth();
        int inHeight = bitmap.getHeight();
        if (inWidth > inHeight) {
            outWidth = maxSize;
            outHeight = (inHeight * maxSize) / inWidth;
        } else {
            outHeight = maxSize;
            outWidth = (inWidth * maxSize) / inHeight;
        }

        return Bitmap.createScaledBitmap(bitmap, outWidth, outHeight, false);
    }

    /**
     * Calculates the sample size.
     *
     * @param options
     * @param reqWidth
     * @param reqHeight
     * @return
     */
    public static int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        // Raw height and width of image
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {

            final int halfHeight = height / 2;
            final int halfWidth = width / 2;

            // Calculate the largest inSampleSize value that is a power of 2 and keeps both
            // height and width larger than the requested height and width.
            while ((halfHeight / inSampleSize) > reqHeight
                    && (halfWidth / inSampleSize) > reqWidth) {
                inSampleSize *= 2;
            }
        }

        return inSampleSize;
    }

    /**
     * Returns a Bitmap as a Base64 string.
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

    /**
     * Returns a Base64 JPEG encoded image.
     *
     * @param bitmap
     * @param quality
     * @return
     */
    public static String getBase64Jpeg(Bitmap bitmap, int quality) {
        return getBase64(bitmap, Bitmap.CompressFormat.JPEG, quality);
    }

    /**
     * Returns the rotation an image was taken with based on the exif parameters.
     *
     * @param fileName
     * @return
     * @throws IOException
     */
    public static int getImageRotation(String fileName) throws IOException {
        ExifInterface exif = new ExifInterface(fileName);
        int rotation;
        switch (exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED)) {
            case ExifInterface.ORIENTATION_NORMAL:
                rotation = 0;
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                rotation = 90;
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                rotation = 180;
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                rotation = 270;
                break;
            default:
                rotation = 0;
                break;
        }

        return rotation;
    }
}
