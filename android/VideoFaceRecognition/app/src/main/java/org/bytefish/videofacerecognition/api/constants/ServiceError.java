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

package org.bytefish.videofacerecognition.api.constants;

import android.util.Log;

public enum ServiceError {

    IMAGE_DECODE_ERROR(10),
    IMAGE_RESIZE_ERROR(11),
    PREDICTION_ERROR(12),
    SERVICE_TEMPORARY_UNAVAILABLE(20),
    UNKNOWN_ERROR(21),
    INVALID_FORMAT(30),
    INVALID_API_KEY(31),
    INVALID_API_TOKEN(32),
    MISSING_ARGUMENTS(40);

    private final int mCode;

    ServiceError(int code) {
        mCode = code;
    }

    public int getCode() {
        return mCode;
    }

    /**
     * Returns a Service Error by its code.
     *
     * @param errorCode
     * @return
     */
    public static ServiceError getServiceErrorByCode(String errorCode) {
        ServiceError serviceError = UNKNOWN_ERROR;
        try {
            Integer errorCodeInt = Integer.parseInt(errorCode);
            for(ServiceError err : ServiceError.values()) {
                if(err.getCode() == errorCodeInt) {
                    serviceError = err;
                    break;
                }
            }
        } catch(NumberFormatException e) {
            Log.e("ServiceError", "Unable to parse given error code: " + errorCode);
        }
        return serviceError;
    }

    /**
     * Returns a human readable error message.
     *
     * @param serviceError
     * @return
     */
    public static String getServiceErrorMessage(ServiceError serviceError) {
        switch(serviceError) {
            case IMAGE_DECODE_ERROR:
                return "Error decoding the image.";
            case IMAGE_RESIZE_ERROR:
                return "Error resizing the image.";
            case PREDICTION_ERROR:
                return "There was an error predicting the image.";
            case SERVICE_TEMPORARY_UNAVAILABLE:
                return "The service is temporarily unavailable.";
            case UNKNOWN_ERROR:
                return "There was an unknown error.";
            case INVALID_FORMAT:
                return "The request has the wrong format.";
            case INVALID_API_KEY:
                return "The given API Key is not valid.";
            case INVALID_API_TOKEN:
                return "The given API Token is invalid.";
            case MISSING_ARGUMENTS:
                return "The request has missing arguments.";
            default:
                return "There was an unknown error.";
        }
    }
}
