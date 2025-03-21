package nhanph.proton.gateway_proton.service.grpc;

import com.nhanph.grpc.SearchFaceRequest;
import com.nhanph.grpc.SearchFaceResponse;
import io.grpc.stub.StreamObserver;
import org.springframework.grpc.server.service.GrpcService;
import com.nhanph.grpc.SearchFaceServiceGrpc;
/**
 * {@code @Package:} nhanph.proton.gateway_proton.service
 * {@code @author:} nhanph
 * {@code @date:} 3/19/2025 2025
 * {@code @Copyright:} @nhanph
 */
@GrpcService
public class SearchFaceServiceImpl extends SearchFaceServiceGrpc.SearchFaceServiceImplBase {

//    @Override
//    public void GetData(SearchFaceRequest searchFaceRequest, StreamObserver<SearchFaceResponse> responseObserver){
//        SearchFaceResponse.Builder responseBuilder = SearchFaceResponse.newBuilder()
//                .setResponseCode("00")
//                .setPeopleId("190903")
//                .setScore(100);
//
//
//    }

    @Override
    public void GetData(SearchFaceRequest searchFaceRequest, SearchFaceResponse searchFaceResponse) {
        SearchFaceResponse.Builder responseBuilder = SearchFaceResponse.newBuilder()
                .setResponseCode("00")
                .setPeopleId("190903")
                .setScore(100);
    }

    @Override
    public void getData(SearchFaceRequest request, StreamObserver<SearchFaceResponse> responseObserver) {
        super.getData(request, responseObserver);
    }
}
