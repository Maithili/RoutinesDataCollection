import random
import json
from math import floor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

personas = []
# personas.append()
activity_map = {
"brush_teeth" : "brushing_teeth",
"bathe_shower" : "showering",
"prepare_eat_breakfast" : "breakfast",
"get_dressed" : "getting_dressed",
"computer_work" : "computer_work",
"prepare_eat_lunch" : "lunch",
"leave_home" : "leaving_home_and_coming_back",
"come_home" : "leaving_home_and_coming_back",
"play_music" : "playing_music",
"read" : "reading",
"take_medication" : "taking_medication",
"prepare_eat_dinner" : "dinner",
"connect_w_friends" : "socializing",
"hand_wash_clothes" : "laundry",
"listen_to_music" : "listening_to_music",
"clean" : "cleaning",
"clean_kitchen" : "kitchen_cleaning",
"take_out_trash" : "take_out_trash",
"do_laundry" : "laundry",
"use_restroom" : "going_to_the_bathroom",
"vacuum_clean" : "vaccuum_cleaning",
"wash_dishes" : "wash_dishes",
"watch_tv" : "watching_tv",
## unnecessary
"diary_journaling" : None,
"wake_up" : None,
"sleep" : None,
}
start_times = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]


class ScheduleDistributionSampler():
    def __init__(self, type):
        activity_histogram = {}
        if type.lower() == 'persona':
            with open('data/personaBasedSchedules/trait_histograms.json') as f:
                trait_histograms = json.load(f)
            persona = random.choice(personas)
            self.label = persona
            for activity in persona:
                activity_histogram[activity] = trait_histograms[activity][persona['activity']]
        elif type.lower() == 'individual':
            with open('data/personaBasedSchedules/individual_histograms.json') as f:
                individual_histograms = json.load(f)
            person = random.choice(list(individual_histograms.keys()))
            self.label = person
            activity_histogram = individual_histograms[person]
        else:
            raise NotImplementedError('Only persona and individual schdule samplers have been implemented')
        
        self.activities = list(activity_histogram.keys())
        self.activity_threshold = np.zeros((len(self.activities)+1, len(start_times)))
        for i, activity in enumerate(self.activities):
            self.activity_threshold[i+1,:] = self.activity_threshold[i,:] + np.array(activity_histogram[activity])
        self.activity_threshold = self.activity_threshold[1:,:]
        self.sampling_range = max(self.activity_threshold[:,-1])
        self.removed_activities = []

    def __call__(self, t_mins, remove=False):
        sample = random.random()*self.sampling_range
        activity = None
        st = start_times.index(int(floor(t_mins/60)))
        for act, thresh in zip(self.activities, list(self.activity_threshold[:,st])):
            if thresh < sample:
                activity = act
                break
        if activity in self.removed_activities:
            return None
        else:
            if remove: self.remove(activity)
            return activity

    def remove(self, activity):
        self.removed_activities.append(activity)

    def plot(self, filepath = None):
        clrs = sns.color_palette("pastel") + sns.color_palette("dark") + sns.color_palette()
        fig, ax = plt.subplots()
        fig.set_size_inches(27, 18.5)
        base = self.activity_threshold[0,:] * 0
        for i, activity in enumerate(self.activities):
            self.activity_threshold[i,:]
            d = self.activity_threshold[i,:] - base
            ax.bar(start_times, d, label=activity, bottom=base, color=clrs[i])
            base = self.activity_threshold[i,:]
        ax.set_xticks(start_times)
        ax.set_xticklabels([str(s)+':00' for s in start_times])
        ax.set_title(self.label)
        plt.legend()
        if filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()

