import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np

from matplotlib import animation
from IPython import display
from IPython.display import clear_output

def plot(frame_idx, rewards, losses, game, game_data):
    ''' Function to plot reward and loss trends. And a frame of the environment'''
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.subplot(133)
    plt.title(f'ep: {game_data[1]} max step: {game_data[2]}')
    plt.imshow(game)
    plt.show()

def plot_reward_solo(rewards, title_game):
    ''' Function to plot only reward trend'''
    plt.figure(figsize=(15,10))
    plt.title('Average Reward on %s' % (title_game))
    plt.plot(rewards)
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    plt.show()

def plot_loss_solo(losses, title_game):
    ''' Function to plot only loss trend'''
    plt.figure(figsize=(15,10))
    plt.title('Average Loss on %s' % (title_game))
    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('Frames')
    plt.show()

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    ''' Function to save a gif from an array of frames'''
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=10)

    anim.save(path + filename, writer='imagemagick', fps=25)