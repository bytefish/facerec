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

import android.util.Log;

import org.apache.http.HttpResponse;
import org.apache.http.HttpStatus;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpRequestBase;
import org.apache.http.client.utils.URLEncodedUtils;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.params.BasicHttpParams;
import org.apache.http.params.HttpConnectionParams;
import org.apache.http.params.HttpParams;
import org.bytefish.videofacerecognition.api.common.Common;
import org.bytefish.videofacerecognition.api.exceptions.AccessDeniedException;
import org.bytefish.videofacerecognition.api.exceptions.InternalServerErrorException;
import org.bytefish.videofacerecognition.api.exceptions.ResourceNotFoundException;
import org.bytefish.videofacerecognition.api.exceptions.RestClientException;
import org.bytefish.videofacerecognition.api.exceptions.WebAppException;
import org.bytefish.videofacerecognition.api.interceptor.LoggingRequestInterceptor;
import org.bytefish.videofacerecognition.api.interceptor.LoggingResponseInterceptor;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.List;
import java.util.Map;

/**
 * This is a Base class for a simple JSON Request/Response scheme.
 *
 * TODO Enable the Client to use SSL
 *
 */
public class BaseServiceClient {

    private static final String TAG = "FaceRecServiceClient";

    // The host of the Web service.
    private final String mHost;
    // HttpClient used for the requests.
    private final DefaultHttpClient mHttpClient;

    public BaseServiceClient(final String host, final String username, final String password) {
        mHost = removeTrailingSlash(host);

        mHttpClient = new DefaultHttpClient();

        setCredentials(username, password);
        setInterceptors();
    }

    public void setInterceptors() {
        // Let's add some logging:
        mHttpClient.addRequestInterceptor(new LoggingRequestInterceptor());
        mHttpClient.addResponseInterceptor(new LoggingResponseInterceptor());
    }

    private void setCredentials(String username, String password) {
        if(username != null && password != null) {
            mHttpClient.getCredentialsProvider().setCredentials(new AuthScope(AuthScope.ANY), new UsernamePasswordCredentials(username, password));
        }
    }

    private String removeLeadingSlash(String str) {
        if(str.startsWith("/")) {
            str = str.substring(1, str.length());
        }
        return str;
    }

    private String removeTrailingSlash(String str) {
        if(str.endsWith("/")) {
            str = str.substring(0, str.length() - 1);
        }
        return str;
    }

    private String getFullUrl(final String relativePath) {
         return mHost + "/" + removeLeadingSlash(relativePath);
    }

    private String getFullUrl(final String relativePath, final List<BasicNameValuePair> parameters) {
        String destUrl = getFullUrl(relativePath);
        String paramString = URLEncodedUtils.format(parameters, "utf-8");
        return destUrl + paramString;
    }

    /**
     * Get a JSON object from the given API method.
     *
     * @param relativePath API method to invoke.
     * @param parameters QueryParameters to add to the request.
     * @return JSON result
     *
     * @throws AccessDeniedException
     * @throws ResourceNotFoundException
     * @throws InternalServerErrorException
     * @throws RestClientException
     */
    protected JSONObject get(String relativePath, List<BasicNameValuePair> parameters)
            throws AccessDeniedException, ResourceNotFoundException, InternalServerErrorException, RestClientException {
        String destUrl = getFullUrl(relativePath, parameters);
        HttpGet httpGet = new HttpGet(destUrl);
        return executeHttpMethod(httpGet);
    }

    /**
     * Post a JSON object to the given API method.
     *
     * @param relativePath API method to request.
     * @param jsonObject JSON data to send to the Webservice.
     * @return JSON result
     *
     * @throws AccessDeniedException
     * @throws ResourceNotFoundException
     * @throws InternalServerErrorException
     * @throws RestClientException
     */
    protected JSONObject post(String relativePath, JSONObject jsonObject)
            throws AccessDeniedException, ResourceNotFoundException, InternalServerErrorException, RestClientException {
        String destUrl = getFullUrl(relativePath);
        HttpPost httpPost = new HttpPost(destUrl);
        if(jsonObject != null) {
            String jsonObjectString = jsonObject.toString();
            try {
                httpPost.setEntity(new StringEntity(jsonObjectString, "UTF-8"));
            } catch (UnsupportedEncodingException e) {
                Log.e(TAG, "Unable to encode JSON data", e);
                throw new RestClientException("Unable to encode the JSON Data", e, jsonObjectString);
            }
            httpPost.setHeader("Accept", "application/json");
            httpPost.setHeader("Content-type", "application/json");
        }
        return executeHttpMethod(httpPost);
    }

    /**
     * This method executes the httpMethod and checks the response. It also evaluates the Status field
     * sent by the Web service.
     *
     * @param httpMethod
     * @return
     * @throws WebAppException
     * @throws AccessDeniedException
     * @throws ResourceNotFoundException
     * @throws InternalServerErrorException
     */
    private JSONObject executeHttpMethod(final HttpRequestBase httpMethod)
            throws AccessDeniedException, ResourceNotFoundException, InternalServerErrorException, RestClientException {

        JSONObject resultJson = null;
        try {
            HttpResponse httpResponse = mHttpClient.execute(httpMethod);
            int httpStatusCode = httpResponse.getStatusLine().getStatusCode();
            if(httpStatusCode != HttpStatus.SC_OK) {
                checkHttpStatus(httpMethod, httpStatusCode);
            }
            resultJson = Common.getJsonFromResponse(httpResponse);
        } catch(IOException e) {
            Log.e(TAG, "Unable to execute HTTP Method.", e);
            throw new RestClientException("Unable executing HTTP Request.", e);
        } catch (JSONException e) {
            Log.e(TAG, "There was an error decoding the JSON Object.", e);
            throw new RestClientException("JSON decoding failed.", e);
        } finally {
            httpMethod.abort();
        }
        return resultJson;
    }

    /**
     * Checks the HttpStatus code of the Response and throws an appropriate exception, the upper
     * layer is able to catch the exception and notify the user.
     *
     * @param httpMethod
     * @param httpStatusCode
     *
     * @throws AccessDeniedException
     * @throws ResourceNotFoundException
     * @throws InternalServerErrorException
     */
    private void checkHttpStatus(HttpRequestBase httpMethod, int httpStatusCode)
            throws AccessDeniedException, ResourceNotFoundException, InternalServerErrorException {
        String requestedPath = httpMethod.getURI().toString();
        if(httpStatusCode == HttpStatus.SC_FORBIDDEN) {
            Log.e(TAG, "Access denied.");
            throw new AccessDeniedException(requestedPath);
        }
        if(httpStatusCode == HttpStatus.SC_NOT_FOUND) {
            Log.e(TAG, "Resource not found.");
            throw new ResourceNotFoundException(requestedPath);
        }
        if(httpStatusCode == HttpStatus.SC_INTERNAL_SERVER_ERROR) {
            Log.e(TAG, "Internal Server Error.");
            throw new InternalServerErrorException(requestedPath);
        }
    }

    /**
     * Sets the timeout to wait for responses.
     *
     * @param milliseconds
     */
    public void setConnectionTimeout(int milliseconds) {
        HttpParams httpParams = mHttpClient.getParams();
        if(httpParams == null) {
            httpParams = new BasicHttpParams();
        }
        HttpConnectionParams.setConnectionTimeout(httpParams, milliseconds);
        HttpConnectionParams.setSoTimeout(httpParams, milliseconds);
        mHttpClient.setParams(httpParams);
    }

    /**
     * Sets this instance to use HTTPS or not.
     *
     * @param enableHttps
     */
    public void setHttps(boolean enableHttps) {
        Log.w(TAG, "HTTPS encryption is not implemented yet.");
    }
}
