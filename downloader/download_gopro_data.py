import argparse
import json
import os
from pathlib import Path
import random

from util import prepare_gopro_2d_output_directory, Constants, download_data

random.seed(42)

def get_balanced_sample_of_videos_to_download(max_number_of_files):

	correct_videos = []
	error_videos = []

	with open(os.path.join(os.path.dirname(__file__), "metadata", "error_annotations.json"), "r") as f:
		error_annotations = json.load(f)
		for annotation in error_annotations:
			if annotation["is_error"]:
				error_videos.append(annotation["recording_id"])
			else:
				correct_videos.append(annotation["recording_id"])
	
	correct_videos = random.sample(correct_videos, max_number_of_files // 2)
	error_videos = random.sample(error_videos, max_number_of_files // 2)
	
	return correct_videos + error_videos


def process_download_gopro_data(download_args):
	# ---- Parse Download Links Json ----
	with open(os.path.join(os.path.dirname(__file__), "metadata", "download_links.json"), "r") as f:
		download_links = json.load(f)

	with open(os.path.join(os.path.dirname(__file__), "metadata", "complete_step_annotations.json"), "r") as f:
		complete_step_annotations = json.load(f)

	video_ids_to_download = get_balanced_sample_of_videos_to_download(download_args.n)
	
	output_dir = Path(download_args.output_dir)
	data_directory = prepare_gopro_2d_output_directory(download_args, output_dir)
	
	download_url_links = []
	download_file_paths = []
	downloaded_video_annotations = []
	for recording_id in video_ids_to_download:
		recording_download_link_dict = download_links[recording_id]
		downloaded_video_annotations.append(complete_step_annotations[recording_id])
		if download_args.data2d:
			if (Constants.GOPRO_RESOLUTION_360P in recording_download_link_dict and
					recording_download_link_dict[Constants.GOPRO_RESOLUTION_360P] is not None):
				gopro_360_url = recording_download_link_dict[Constants.GOPRO_RESOLUTION_360P]
				gopro_360p_path = data_directory / Constants.GOPRO / Constants.RESOLUTION_360P / f"{recording_id}_360p.mp4"
				download_url_links.append(gopro_360_url)
				download_file_paths.append(gopro_360p_path)
			else:
				if recording_download_link_dict[Constants.HOLOLENS_SYNC_PV_VIDEO] is not None:
					hololens_pv_url = recording_download_link_dict[Constants.HOLOLENS_SYNC_PV_VIDEO]
					hololens_pv_path = data_directory / Constants.GOPRO / Constants.RESOLUTION_360P / f"{recording_id}_360p.mp4"
					download_url_links.append(hololens_pv_url)
					download_file_paths.append(hololens_pv_path)
					print(f"Hololens 360P data downloaded for {recording_id}")
			
			if download_args.resolution4K:
				if recording_download_link_dict[Constants.GOPRO_RESOLUTION_4K] is not None:
					gopro_4k_url = recording_download_link_dict[Constants.GOPRO_RESOLUTION_4K]
					gopro_4k_path = data_directory / Constants.GOPRO / Constants.RESOLUTION_4K / f"{recording_id}_4K.mp4"
					download_url_links.append(gopro_4k_url)
					download_file_paths.append(gopro_4k_path)
	
	print("-------------------------------------------------")
	print(f"Downloading {len(download_url_links)} files")
	download_data(download_url_links, download_file_paths)

	download_path_parent_dir = os.path.dirname(download_file_paths[0])
	downloaded_video_annotations_path = os.path.join(download_path_parent_dir,  "downloaded_video_annotations.json")
	with open(downloaded_video_annotations_path, "w") as f:
		json.dump(downloaded_video_annotations, f)

	return downloaded_video_annotations_path


if __name__ == "__main__":
	print("Starting the download process")
	
	# Create the parser
	parser = argparse.ArgumentParser(description='Download the data from BOX Cloud')
	
	parser.add_argument('--data2d', action='store_true',
	                    help='Use this to download 2D data from Box Cloud which includes GOPRO [360p] data')
	parser.add_argument('--resolution4K', action='store_true',
	                    help='Use this to default download 4K data from Box Cloud which includes GOPRO [4K] data')
	
	parser.add_argument('--output_dir', type=str, default="./", help='Output directory to store the downloaded data')

	parser.add_argument("--n", type=int, default=None, help='Number of files to download')
	
	# Parse the arguments
	args = parser.parse_args()
	
	downloaded_video_annotations_path = process_download_gopro_data(args)
	print(f"Downloaded video annotations path: {downloaded_video_annotations_path}")
