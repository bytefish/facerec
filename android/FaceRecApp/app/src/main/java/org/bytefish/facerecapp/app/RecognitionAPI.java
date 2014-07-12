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
import android.util.Log;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.StatusLine;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.DefaultHttpClient;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * This class is just a simple wrapper for the API defined in the blog
 * article. Basically all it does is creating a JSON Object to request
 * the recognition and waits (synchronously) for the result.
 *
 *
 */
public class RecognitionAPI {

    private static final String TAG = RecognitionAPI.class.getName();

    private static final String RECOGNITION_API_URL = "http://192.168.178.21:5050/predict";

    public static String requestRecognition(Bitmap bitmap) throws Exception {
        String result = new String();
        try {
            JSONObject jsonRequest = constructPredictionRequest(bitmap);
            HttpResponse response = makeSynchronousRequest(jsonRequest);
            result = evaluateResponse(response);
        } catch (Exception e) {
            Log.e(TAG, "Recognition failed!", e);
            throw e;
        }
        return result;
    }

    private static HttpResponse makeSynchronousRequest(JSONObject jsonRequest) throws IOException {
        DefaultHttpClient httpClient = new DefaultHttpClient();
        HttpPost httpPost = new HttpPost(RECOGNITION_API_URL);
        httpPost.setEntity(new StringEntity(jsonRequest.toString()));
        httpPost.setHeader("Accept", "application/json");
        httpPost.setHeader("Content-type", "application/json");

        return httpClient.execute(httpPost);
    }

    private static String evaluateResponse(HttpResponse response) throws IOException, JSONException {
        String result = new String();
        if (response.getStatusLine().getStatusCode() == 200) {
            HttpEntity entity = response.getEntity();
            String entityContent = readEntityContentToString(entity);
            JSONObject jsonResponse = new JSONObject(entityContent);
            result = jsonResponse.getString("name");
        }
        return result;
    }

    private static String readEntityContentToString(HttpEntity entity) throws IOException {
        StringBuilder result = new StringBuilder();
        InputStream content = entity.getContent();
        BufferedReader reader = new BufferedReader(new InputStreamReader(content));
        String line;
        while ((line = reader.readLine()) != null) {
            result.append(line);
        }

        return result.toString();
    }

    private static JSONObject constructPredictionRequest(Bitmap bitmap) throws JSONException {
        String base64encBitmap = ImageHelper.getBase64Jpeg(bitmap, 100);
        // Now create the object:
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("image", base64encBitmap);
        } catch (JSONException e) {
            Log.e(TAG, "Could not create Json object", e);
            throw e;
        }
        return jsonObject;
    }
}
