import threading
import fastdds
import RGBData


class _RGBDataListener(fastdds.DataReaderListener):
    """内部监听器，用于更新最新数据"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def on_data_available(self, reader):
        try:
            info = fastdds.SampleInfo()
            data = RGBData.RGBData()
            #while reader.take_next_sample(data, info) == fastdds.ReturnCode_t.RETCODE_OK:
            while reader.take_next_sample(data, info) == fastdds.RETCODE_OK:
                if info.valid_data:
                    # 只保存简单的原始数据（不拷贝 SWIG 对象）
                    with self.parent._lock:
                        self.parent._latest_data_raw = {
                            'width': data.width(),
                            'height': data.height(),
                            'data_ptr': data.data()  # 你也可以根据需求提取 numpy 数组
                        }
                        self.parent._message_count += 1
        except Exception as e:
            print("Listener 异常:", e)

class RGBDataSubscriber:
    """FastDDS RGBData 订阅封装类"""
    def __init__(self, topic_name="RgbTopic", domain_id=0):
        self.topic_name = topic_name
        self.domain_id = domain_id
        self._lock = threading.Lock()
        self._latest_data_raw = None
        self._message_count = 0

        # DomainParticipant
        factory = fastdds.DomainParticipantFactory.get_instance()
        self.participant_qos = fastdds.DomainParticipantQos()
        factory.get_default_participant_qos(self.participant_qos)
        self.participant = factory.create_participant(self.domain_id, self.participant_qos)
        if self.participant is None:
            raise RuntimeError("创建 DomainParticipant 失败")

        # Type注册
        self.topic_data_type = RGBData.RGBDataPubSubType()
        self.topic_data_type.set_name("RGBData")
        self.type_support = fastdds.TypeSupport(self.topic_data_type)
        self.participant.register_type(self.type_support)

        # Topic
        self.topic_qos = fastdds.TopicQos()
        self.participant.get_default_topic_qos(self.topic_qos)
        self.topic = self.participant.create_topic(self.topic_name, self.topic_data_type.get_name(), self.topic_qos)
        if self.topic is None:
            raise RuntimeError("创建 Topic 失败")

        # Subscriber
        self.subscriber_qos = fastdds.SubscriberQos()
        self.participant.get_default_subscriber_qos(self.subscriber_qos)
        self.subscriber = self.participant.create_subscriber(self.subscriber_qos)
        if self.subscriber is None:
            raise RuntimeError("创建 Subscriber 失败")

        # Listener & DataReader
        self.listener = _RGBDataListener(self)
        self.reader_qos = fastdds.DataReaderQos()
        self.subscriber.get_default_datareader_qos(self.reader_qos)
        self.reader = self.subscriber.create_datareader(self.topic, self.reader_qos, self.listener)
        if self.reader is None:
            raise RuntimeError("创建 DataReader 失败")

        print(f"✅ FastDDS RGBDataSubscriber 已启动，监听话题: {self.topic_name}")

    def get_latest(self):
        """返回最新一帧 RGBData 原始信息（线程安全）"""
        with self._lock:
            return self._latest_data_raw

    def get_message_count(self):
        """返回接收消息总数"""
        with self._lock:
            return self._message_count

    def close(self):
        """释放 DDS 资源"""
        print("🧹 释放 FastDDS 资源...")
        if self.subscriber and self.reader:
            self.subscriber.delete_datareader(self.reader)
        if self.participant and self.subscriber:
            self.participant.delete_subscriber(self.subscriber)
        if self.participant and self.topic:
            self.participant.delete_topic(self.topic)
        if self.participant:
            factory = fastdds.DomainParticipantFactory.get_instance()
            factory.delete_participant(self.participant)
        print("✅ 已安全关闭。")

