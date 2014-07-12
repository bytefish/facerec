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
import android.media.FaceDetector;

/**
 * Wraps up some FaceDetection related methods.
 */
public class FaceDetectionHelper {

    private static final int MAX_FACES = 5;

    private static void makeImageWidthEven(Bitmap image) {
        int width = image.getWidth();
        int height = image.getHeight();
        // Make width of the image even:
        if(width % 2 != 0) {
            image = ImageHelper.cropImage(image, 0, 0, width - 1, height);
        }
    }

    public static FaceDetector.Face[] detectFaces(Bitmap image) {
        // We need an even image for face detection:
        makeImageWidthEven(image);
        // Try to detect faces:
        FaceDetector.Face[] facesInImage = new FaceDetector.Face[MAX_FACES];
        FaceDetector faceDetector = new FaceDetector(image.getWidth(), image.getHeight(), MAX_FACES);
        int numFaces = faceDetector.findFaces(image, facesInImage);
        // Only return the faces we have found:
        FaceDetector.Face[] result = new FaceDetector.Face[numFaces];
        for(int pos = 0; pos < numFaces; pos++) {
            result[pos] = facesInImage[pos];
        }
        return result;
    }
}
