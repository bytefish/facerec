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

package org.bytefish.videofacerecognition.api.client;

import android.graphics.Bitmap;
import android.util.Log;

import org.bytefish.videofacerecognition.api.constants.Constants;
import org.bytefish.videofacerecognition.api.exceptions.AccessDeniedException;
import org.bytefish.videofacerecognition.api.exceptions.InternalServerErrorException;
import org.bytefish.videofacerecognition.api.exceptions.ResourceNotFoundException;
import org.bytefish.videofacerecognition.api.exceptions.RestClientException;
import org.bytefish.videofacerecognition.api.exceptions.WebAppException;
import org.bytefish.videofacerecognition.app.util.Util;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Uses the BaseServiceClient to implement the RESTful API used for the Flask facerec service.
 *
 */
public class FaceRecServiceClient {

    private static final String TAG = "FaceRecServiceClient";

    private BaseServiceClient mBaseServiceClient;

    public static final String API_RECOGNIZE = "api/recognize";

    public FaceRecServiceClient(final String host, final String username, final String password) {
        mBaseServiceClient = new BaseServiceClient(host, username, password);
    }

    /**
     * Makes a Recognition request to the Server.
     *
     * @param bitmap
     * @return
     * @throws AccessDeniedException
     * @throws ResourceNotFoundException
     * @throws InternalServerErrorException
     * @throws WebAppException
     * @throws RestClientException
     */
    public String recognize(Bitmap bitmap)
            throws AccessDeniedException, ResourceNotFoundException, InternalServerErrorException, WebAppException, RestClientException {
        JSONObject requestObject = null;

        try {
            requestObject = buildRecognizeRequest(bitmap);
        } catch(JSONException e) {
            throw new RestClientException("Unable to parse JSON data", e);
        }
        JSONObject responseObject = post(API_RECOGNIZE, requestObject);

        return responseObject.optString("name");
    }

    /**
     * Builds the JSON Object for the Recognition request, by turning a Bitmap into its
     * Base64 representation.
     *
     * @param bitmap
     * @return
     * @throws JSONException
     */
    private JSONObject buildRecognizeRequest(Bitmap bitmap) throws JSONException {
        JSONObject requestObject = new JSONObject();
        String bitmapBase64 = Util.getBase64(bitmap, Bitmap.CompressFormat.JPEG, 100);
        requestObject.put("image", bitmapBase64);
        return requestObject;
    }

    /**
     *
     * @param relativePath
     * @param jsonObject
     * @return
     * @throws AccessDeniedException
     * @throws ResourceNotFoundException
     * @throws InternalServerErrorException
     * @throws RestClientException
     * @throws WebAppException
     */
    private JSONObject post(String relativePath, JSONObject jsonObject) throws AccessDeniedException, ResourceNotFoundException, InternalServerErrorException, RestClientException, WebAppException {
        JSONObject resultJson = mBaseServiceClient.post(relativePath, jsonObject);
        if(resultJson != null) {
            evaluateStatus(resultJson);
        }
        return resultJson;
    }

    /**
     * Evaluates the status field of the Web service response, so we can throw a
     * WebAppException if necessary.
     *
     * @param jsonObject
     * @throws WebAppException
     */
    private void evaluateStatus(JSONObject jsonObject) throws WebAppException {
        String status = jsonObject.optString(Constants.STATUS_KEY);
        if (Constants.STATUS_ERROR.equals(status)) {
            String errorCode = jsonObject.optString("code");
            String errorMessage = jsonObject.optString("message");

            final WebAppException e = new WebAppException(errorCode, errorMessage);
            Log.e(TAG, errorMessage, e);
            throw e;
        }
    }

    public void setConnectionTimeout(int seconds) {
        int milliseconds = seconds * 1000;
        mBaseServiceClient.setConnectionTimeout(milliseconds);
    }

    public void setHttps(boolean enableHttps) {
        mBaseServiceClient.setHttps(enableHttps);
    }
}
