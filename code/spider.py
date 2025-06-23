import pandas as pd
import matplotlib.pyplot as plt
import imageio
from moviepy.editor import *

""" output_dir = '/Users/kaiali/Documents/HCP/NEOM/mat/output/spider_plot/mp4'
df = pd.read_csv('/Users/kaiali/Documents/HCP/NEOM/mat/output/matrix/average_trans.csv', header=0)
print(df.head(2))

# Define a function to plot a spider chart for a row of the dataframe
def plot_spider(row):
    categories = list(df.columns)
    values = row.values.tolist()
    values += values[:1]
    angles = [n / float(len(categories)) * 2 * 3.141592653589793 for n in range(len(categories))]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.set_rlabel_position(0)
    plt.yticks([0.05,0.1,0.2,0.25,0.4,0.5], ["0.05","0.1","0.2","0.25","0.4","0.5"], color="grey", size=7)
    plt.ylim(0,0.5)
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

# Calculate the frame rate based on the desired time per frame
time_per_frame = 1.6 # seconds
frame_rate = 1 / time_per_frame

# Loop through each row of the dataframe and plot the spider chart
filenames = []
for i in range(len(df)):
    plot_spider(df.iloc[i])
    plt.title("Time point {}".format(i))
    filename = os.path.join(output_dir,"plot_{}.png".format(i))
    filenames.append(filename)
    plt.savefig(filename)
    plt.clf()

# Combine the video and the spider chart gif into a single animation
video = VideoFileClip('/Users/kaiali/Documents/HCP/NEOM/mat/test_230215_updated_vid1.mp4')
gif = ImageSequenceClip(filenames, durations=[time_per_frame] * len(filenames)).resize(height=video.h, width=video.w)

# Set the duration of the gif to match the duration of the video
gif = gif.set_duration(video.duration)

# Create a CompositeVideoClip to overlay the gif onto the video
# final_animation = CompositeVideoClip([video, gif.set_position(('right', 0))])
final_animation = CompositeVideoClip([video.set_position((0, 0)),
                                      gif.set_position(('right', 0))],
                                     size=(video.w + gif.w, max(video.h, gif.h)))

# Save the result as an mp4 file
# final_animation.write_gif('/Users/kaiali/Documents/HCP/NEOM/mat/output/spider_plot/avg_t.gif')

final_animation.write_videofile('/Users/kaiali/Documents/HCP/NEOM/mat/output/spider_plot/mp4/avg_20.mp4', codec='libx264', audio_codec="aac")
 """


import pandas as pd
import matplotlib.pyplot as plt
import imageio
import numpy as np
from scipy.interpolate import interp1d
from moviepy.editor import *

output_dir = '/Users/kaiali/Documents/HCP/NEOM/mat/v3/spider_plot'
df = pd.read_csv('/Users/kaiali/Documents/HCP/NEOM/mat/v3/matrix/new_average_trans_.csv', header=0)
print(df.head(2))

desired_frames = 2733



# Combine the video and the spider chart gif into a single animation
video = VideoFileClip('/Users/kaiali/Documents/HCP/NEOM/mat/v3/Video3.mp4')
time_per_frame = video.duration / desired_frames


image_folder_1 = '/Users/kaiali/Documents/HCP/NEOM/mat/v3/spider_plot'
gif_path_1 = '/Users/kaiali/Documents/HCP/NEOM/mat/v3/gif1.gif'
mp4_path_1 = '/Users/kaiali/Documents/HCP/NEOM/mat/v3/spider.mp4'

def images_to_gif(image_folder_1, gif_path_1, duration=[time_per_frame] * 2733):#len(filenames1)
    image_filenames = sorted([filename for filename in os.listdir(image_folder_1) if not filename.startswith('.')])
    image_paths = [os.path.join(image_folder_1, filename) for filename in image_filenames]
    images = []
    for path in image_paths:
        images.append(imageio.imread(path))
    imageio.mimsave(gif_path_1, images, duration=duration)

gif1 = images_to_gif(image_folder_1, gif_path_1, duration=[time_per_frame] * 2733)#len(filenames1)

video_1 = video.resize(height=video.h // 2, width=video.w)

video_1.write_videofile(mp4_path_1, codec='libx264')







image_folder_2 = '/Users/kaiali/Documents/HCP/NEOM/mat/v3/MMP_3D_image_0'
gif_path_2 = '/Users/kaiali/Documents/HCP/NEOM/mat/v3/gif2.gif'
mp4_path_2 = '/Users/kaiali/Documents/HCP/NEOM/mat/v3/b3.mp4'

def images_to_gif(image_folder_2, gif_path, duration=[time_per_frame] * 2733):#len(filenames1)
    image_filenames = sorted([filename for filename in os.listdir(image_folder_2) if not filename.startswith('.')])
    image_paths = [os.path.join(image_folder_2, filename) for filename in image_filenames]
    images = []
    for path in image_paths:
        images.append(imageio.imread(path))
    imageio.mimsave(gif_path, images, duration=duration)

gif2 = images_to_gif(image_folder_2, gif_path_2, duration=[time_per_frame] * 2733)#len(filenames1)
video_2 = video.resize(height=video.h // 2, width=video.w)

video_2.write_videofile(mp4_path_2, codec='libx264')




# Create a CompositeVideoClip to overlay the gif onto the video
final_animation = CompositeVideoClip([#video.set_position((0, 0)),
                                      #gif.set_position(('right', 0))],
                                      gif1.set_position(('left', 'bottom')),
                                      gif2.set_position(('right', 'bottom')),
                                      video.set_position(('center', 'top'))],
                                     size=(max(video.w , gif1.w), video.h + gif1.h))

# Save the result as an mp4
final_animation.write_videofile('/Users/kaiali/Documents/HCP/NEOM/mat/v3/spider_plot/avg.mp4',
                                codec='libx264', audio_codec="aac", remove_temp=False)
#, fps=1 / time_per_frame
