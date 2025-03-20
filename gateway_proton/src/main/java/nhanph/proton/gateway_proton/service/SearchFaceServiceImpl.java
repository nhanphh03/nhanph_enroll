package nhanph.proton.gateway_proton.service;

import com.nhanph.grpc.SearchFaceRequest;
import com.nhanph.grpc.SearchFaceResponse;
import com.nhanph.grpc.SearchFaceServiceGrpc;
import io.grpc.stub.StreamObserver;
import org.springframework.grpc.server.service.GrpcService;

/**
 * {@code @Package:} nhanph.proton.gateway_proton.service
 * {@code @author:} nhanph
 * {@code @date:} 3/19/2025 2025
 * {@code @Copyright:} @nhanph
 */
@GrpcService
public class SearchFaceServiceImpl extends SearchFaceServiceGrpc.SearchFaceServiceImplBase {

    @Override
    public void getData(SearchFaceRequest request, StreamObserver<SearchFaceResponse> responseObserver) {
        // Xử lý logic tại đây
        SearchFaceResponse response = SearchFaceResponse.newBuilder()
                .setResponseCode("200")
                .setPeopleId("12345")
                .setScore(95)
                .build();

        // Gửi phản hồi về client
        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }
}
