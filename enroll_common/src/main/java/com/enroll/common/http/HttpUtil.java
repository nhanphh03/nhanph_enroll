package com.enroll.common.http;

import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.util.EntityUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.DeserializationFeature;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;

import static com.enroll.common.constant.Constant.APPLICATION_JSON;
import static com.enroll.common.constant.Constant.CONTENT_TYPE;

/**
 * {@code @Package:} nhanph.proton.gateway_proton.config
 * {@code @author:} nhanph
 * {@code @date:} 3/20/2025
 * {@code @Copyright:} @nhanph
 */

@Slf4j
@Component
public class HttpUtil {

    private final CloseableHttpClient httpClient;
    private final ObjectMapper objectMapper;

    public HttpUtil() {
        PoolingHttpClientConnectionManager connectionManager = new PoolingHttpClientConnectionManager();
        connectionManager.setMaxTotal(100);
        connectionManager.setDefaultMaxPerRoute(20);

        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectionRequestTimeout(5000)
                .setConnectTimeout(5000)
                .setSocketTimeout(5000)
                .build();

        this.httpClient = HttpClients.custom()
                .setConnectionManager(connectionManager)
                .setDefaultRequestConfig(requestConfig)
                .build();

        this.objectMapper = new ObjectMapper();
        this.objectMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    }

    /**
     * Gửi request POST đến một URL với request body và nhận phản hồi kiểu dữ liệu mong muốn.
     *
     * @param url          URL endpoint
     * @param requestId    ID của request để log
     * @param requestBody  Dữ liệu gửi đi
     * @param responseType Kiểu dữ liệu phản hồi mong muốn
     * @param <T>          Kiểu dữ liệu trả về
     * @return Đối tượng phản hồi đã parse
     */
    public <T> T sendPostRequest(String url, String requestId,Object requestBody, Class<T> responseType) {
        HttpPost httpPost = new HttpPost(url);
        httpPost.setHeader(CONTENT_TYPE, APPLICATION_JSON);
        try {
            String jsonRequest = objectMapper.writeValueAsString(requestBody);
            StringEntity entity = new StringEntity(jsonRequest, StandardCharsets.UTF_8);
            httpPost.setEntity(entity);
            log.debug("Request: {} ---- {} ---- {}", requestId, url, requestBody.toString());
            try (CloseableHttpResponse response = httpClient.execute(httpPost)) {
                HttpEntity responseEntity = response.getEntity();
                if (responseEntity != null) {
                    String jsonResponse = EntityUtils.toString(responseEntity, StandardCharsets.UTF_8);
                    log.debug("Response: {} --- {} ---- {}", requestId, url, jsonResponse);
                    return objectMapper.readValue(jsonResponse, responseType);
                }
            }
        } catch (Exception e) {
            log.error("Request: {} ----- Error while sending POST request to  ---- {} ", requestId, url, e);
            throw new RuntimeException("Error while sending POST request to " + url, e);
        }
        return null;
    }
}