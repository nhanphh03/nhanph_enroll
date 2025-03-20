package nhanph.proton.gateway_proton.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import lombok.Getter;

/**
 * {@code @Package:} nhanph.proton.gateway_proton.config
 * {@code @author:} nhanph
 * {@code @date:} 3/19/2025
 * {@code @Copyright:} @nhanph
 */
@Configuration
@Getter
public class ApiConfig {

    @Value("${url.face.search}")
    private String urlSearch;

    @Value("${url.face.register}")
    private String urlRegister;

    @Value("${url.face.remove}")
    private String urlRemove;
}
