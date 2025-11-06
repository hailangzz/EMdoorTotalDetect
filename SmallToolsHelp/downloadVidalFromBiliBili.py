
import os

import subprocess





def download_video(url,save_path="/home/chenkejing/Videos/test_videos"):

    try:

        # 调用you-get命令下载视频，-o指定下载到当前目录

        subprocess.run(['you-get', '-o', save_path, url], check=True)

        print('视频下载成功！')

    except subprocess.CalledProcessError as e:

        print(f'下载失败: {e}')





if __name__ == "__main__":

    # video_url = input('请输入视频网页链接: ')
    video_url = r"https://www.bilibili.com/video/BV1uD4y1k781/?spm_id_from=333.337.search-card.all.click&vd_source=c40f53901650851dbd7776342a33a5ee"

    download_video(video_url)

print("download finish!!!")