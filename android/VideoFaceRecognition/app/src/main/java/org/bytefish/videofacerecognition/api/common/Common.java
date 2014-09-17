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

package org.bytefish.videofacerecognition.api.common;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.util.EntityUtils;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Common methods used for all HTTP API requests. This may include signatures in future iterations,
 * or other security measures for the Web service.
 */
public class Common {

    public static final String CONTENT_TYPE_JSON = "application/json";

    public static boolean isNullOrWhiteSpace(String str) {
        if(str == null || str.isEmpty()) {
            return true;
        }
        for(char c : str.toCharArray()) {
            if(!Character.isWhitespace(c)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Turns the response of an HTTP request into a JSON Object.
     *
     * @param response
     * @return
     * @throws IOException
     * @throws JSONException
     */
    public static JSONObject getJsonFromResponse(HttpResponse response) throws IOException, JSONException {
        HttpEntity entity = response.getEntity();
        String entityContent = EntityUtils.toString(entity);
        return new JSONObject(entityContent);
    }

    //region Signature Utilities

    final protected static char[] hexArray = "0123456789ABCDEF".toCharArray();

    /**
     * Computes the SHA1 Hash of given UTF-8 data.
     *
     * @param message
     * @return
     * @throws UnsupportedEncodingException
     * @throws NoSuchAlgorithmException
     */
    public static String SHA1(String message) throws UnsupportedEncodingException, NoSuchAlgorithmException
    {
        byte[] data = message.getBytes("UTF-8");
        MessageDigest md = MessageDigest.getInstance("SHA1");
        md.update(data, 0, data.length);
        byte[] digest = md.digest();
        return bytesToHex(digest);
    }

    /**
     * Converts a Byte Array into a Hexadecimal String representation, this method
     * is taken from: http://stackoverflow.com/questions/9655181/convert-from-byte-array-to-hex-string-in-java
     *
     * @param bytes
     * @return
     */
    public static String bytesToHex(byte[] bytes) {
        char[] hexChars = new char[bytes.length * 2];
        for ( int j = 0; j < bytes.length; j++ ) {
            int v = bytes[j] & 0xFF;
            hexChars[j * 2] = hexArray[v >>> 4];
            hexChars[j * 2 + 1] = hexArray[v & 0x0F];
        }
        return new String(hexChars);
    }

    //endregion
}
