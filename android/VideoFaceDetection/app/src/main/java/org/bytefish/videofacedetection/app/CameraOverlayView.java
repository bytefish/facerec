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

package org.bytefish.videofacedetection.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.hardware.Camera.Face;
import android.view.Surface;
import android.view.View;


public class CameraOverlayView extends View {

    private Paint mPaint;
    private int mRotation;
    private Face[] mFaces;

    /**
     * Face coordinates are given in a (-1000,1000) coordinate system. We need
     * to normalize them to a value between (0,width) and (0,height).
     */
    public static final float COORDINATE_NORMALIZE = 2000;

    public CameraOverlayView(Context context) {
        super(context);
        initialize();
    }

    private void initialize() {
        // We want a green box around the face:
        mPaint = new Paint();
        mPaint.setAntiAlias(true);
        mPaint.setDither(true);
        mPaint.setColor(Color.GREEN);
        mPaint.setAlpha(128);
        mPaint.setStyle(Paint.Style.FILL_AND_STROKE);
    }

    public void setFaces(Face[] faces) {
        mFaces = faces;
        invalidate();
    }

    public void setRotation(int rotation) {
        mRotation = rotation;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (mFaces == null) {
            return;
        }
        for (Face face : mFaces) {
            if (face == null) {
                continue;
            }
            // Create the matrix for
            Matrix matrix = new Matrix();
            // Rotate the image according to the current screen orientation:
            switch (mRotation) {
                case Surface.ROTATION_0:
                    matrix.postRotate(90);
                    break;
                case Surface.ROTATION_90:
                    matrix.postRotate(0);
                    break;
                case Surface.ROTATION_180:
                    matrix.postRotate(270);
                    break;
                case Surface.ROTATION_270:
                    matrix.postRotate(180);
                    break;
                default:
                    matrix.postRotate(0);
                    break;
            }
            // Normalize the coordinates:
            matrix.postScale(getWidth() / COORDINATE_NORMALIZE, getHeight() / COORDINATE_NORMALIZE);
            matrix.postTranslate(getWidth() / 2f, getHeight() / 2f);
            int saveCount = canvas.save();
            canvas.concat(matrix);
            canvas.drawRect(face.rect, mPaint);
            canvas.restoreToCount(saveCount);
        }
    }

}