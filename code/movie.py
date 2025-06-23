best_train_loss = float('inf')  # initialize with positive infinity
best_val_loss = None

for epoch in range(num_epochs):
    # train for one epoch
    train_loss = train(model, train_loader, optimizer, criterion)
    
    # evaluate on validation set
    with torch.no_grad():
        val_loss = evaluate(model, val_loader, criterion)
    
    # update best train and val losses
    if val_loss < best_loss:
            best_loss = val_loss
            best_model = projector.state_dict()
            best_fold = fold_idx








# Define number of folds
n_folds = 5

# Define cross-validation iterator
cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Define model parameters
input_dim = X_train.shape[1]
hidden_dims = 128
n_layers = 2
mlp_ratio = 2
lr = 1e-3
n_epochs = 100

# Define criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Create lists to store train and val losses for each fold
train_losses = []
val_losses = []

# Loop over the folds
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
    print(f"Fold {fold+1}/{n_folds}")
    # Split data into training and validation sets for this fold
    X_train_fold = torch.tensor(X_train[train_idx], dtype=torch.float32)
    y_train_fold = torch.tensor(y_train[train_idx], dtype=torch.float32)
    X_val_fold = torch.tensor(X_train[val_idx], dtype=torch.float32)
    y_val_fold = torch.tensor(y_train[val_idx], dtype=torch.float32)
    
    # Create model instance
    model = get_mlp(input_dim=input_dim, hidden_dims=hidden_dims, n_layers=n_layers, mlp_ratio=mlp_ratio)
    model.to(device)
    
    # Train model for this fold
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    for epoch in range(n_epochs):
        # Train model
        train_loss = train(model, criterion, optimizer, X_train_fold, y_train_fold)
        
        # Compute validation loss
        val_loss = evaluate(model, criterion, X_val_fold, y_val_fold)
        
        # Update best loss and save model if current validation loss is lower
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            torch.save(model.state_dict(), f"best_model_fold{fold+1}.pt")
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Print best train and val losses for this fold
    print(f"Best train loss for fold {fold+1}: {best_train_loss:.4f}")
    print(f"Best val loss for fold {fold+1}: {best_val_loss:.4f}")
    
    # Save train and val losses for this fold
    train_losses.append(best_train_loss)
    val_losses.append(best_val_loss)

# Compute and print average val loss
avg_val_loss = sum(val_losses) / n_folds
print(f"Average val loss: {avg_val_loss:.4f}")





















import pandas as pd
import matplotlib.pyplot as plt
import imageio
import numpy as np
from scipy.interpolate import interp1d
from moviepy.editor import *

output_dir = '/Users/kaiali/Documents/HCP/NEOM/mat/output/spider_plot'
df = pd.read_csv('/Users/kaiali/Documents/HCP/NEOM/mat/output/matrix/average_trans.csv', header=0)
print(df.head(2))

# Define a function to plot a spider chart for a row of the dataframe
def plot_spider(row):
    categories = list(df.columns)
    values = row.values.tolist()
    values += values[:1]
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.set_rlabel_position(0)
    plt.yticks([0.05,0.15,0.25,0.4,0.5], ["0.05","0.15","0.25","0.4","0.5"], color="grey", size=7)
    plt.ylim(0,0.5)
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

# Calculate the upsampling factor based on the desired number of frames
desired_frames = 1000
upsampling_factor = desired_frames / len(df)

# Perform upsampling of the dataframe
upsampled_df = df.copy()
upsampled_df.index *= upsampling_factor
upsampled_df = upsampled_df.reindex(np.arange(0, desired_frames), method='nearest')

# Loop through each row of the upsampled dataframe and plot the spider chart
filenames = []
for i in range(len(upsampled_df)):
    plot_spider(upsampled_df.iloc[i])
    plt.title("Time point {}".format(i))
    filename = os.path.join(output_dir,"plot_{}.png".format(i))
    filenames.append(filename)
    plt.savefig(filename)
    plt.clf()

# Combine the video and the spider chart gif into a single animation
video = VideoFileClip('/Users/kaiali/Documents/HCP/NEOM/mat/test_230215_updated_vid1.mp4')
time_per_frame = video.duration / desired_frames
gif = ImageSequenceClip(filenames, durations=[time_per_frame] * len(filenames)).resize(height=video.h, width=video.w)

# Create a CompositeVideoClip to overlay the gif onto the video
final_animation = CompositeVideoClip([video.set_position((0, 0)),
                                      gif.set_position(('right', 0))],
                                     size=(video.w + gif.w, max(video.h, gif.h)))

# Save the result as an mp4 file
final_animation.write_videofile('/Users/kaiali/Documents/HCP/NEOM/mat/output/spider_plot/avg_test.mp4',
                                codec='libx264', audio_codec="aac", fps=1 / time_per_frame, remove_temp=False)
