package com.enroll.common.constant;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/**
 * {@code @Package:} nhanph.proton.gateway_proton.config
 * {@code @author:} nhanph
 * {@code @date:} 3/20/2025
 * {@code @Copyright:} @nhanph
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class Constant {
    public static class ResponseCode{
        public static final int OK = 200;
        public static final int CREATED = 201;
        public static final int ACCEPTED = 202;
        public static final int BAD_REQUEST = 400;
        public static final int FORBIDDEN = 403;
        public static final int NOT_FOUND = 404;
        public static final int METHOD_NOT_ALLOWED = 405;
        public static final int CONFLICT = 409;
        public static final int INTERNAL_SERVER_ERROR = 500 ;
        public static final int NOT_IMPLEMENTED = 501 ;
        public static final int  BAD_GATEWAY = 502;
        public static final int  SERVICE_UNAVAILABLE = 503;
        public static final int  GATEWAY_TIMEOUT = 504;
    }

    public static final String APPLICATION_JSON = "application/json";
    public static final String CONTENT_TYPE = "Content-Type";
}
