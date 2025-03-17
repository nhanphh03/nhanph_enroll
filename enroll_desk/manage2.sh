#!/bin/sh

while [ true ]; do
    d=$(date)
    echo "Start $d"

    # Kill tiến trình cũ đúng cách
    pgrep -f "face-enroll-app.jar" | xargs kill -9
    sleep 5s  

    cd /app/face-enroll-app-v2.3.4

    # Thiết lập biến môi trường
    export GDK_BACKEND=x11
    export JAVA_FX_PRISM_ORDER=sw

    # Chạy ứng dụng và lưu log
    ./jdk/bin/java \
        -Xms256M \
        -Xmx4096M \
        -XX:MaxMetaspaceSize=512M \
        -XX:+HeapDumpOnOutOfMemoryError \
        -XX:HeapDumpPath=./dump/heapdump.bin \
        -Dfile.encoding=UTF-8 \
        -Djava.library.path="/home/chamcong/Downloads/opencv-3.4.12/build/lib" \
        -Dprism.order=sw \
        --module-path ./javafx-sdk/lib \
        --add-modules javafx.controls,javafx.fxml \
        -jar face-enroll-app.jar >> /var/log/face-enroll.log 2>&1 &

    APP_PID=$!
    echo "Ứng dụng đang chạy với PID: $APP_PID"

    # Kiểm tra nếu ứng dụng bị dừng thì restart
    while kill -0 $APP_PID 2>/dev/null; do
        sleep 10
    done

    echo "Ứng dụng đã dừng! Restarting..."
    sleep 5s
done









 




