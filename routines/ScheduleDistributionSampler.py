import random
import json
from math import floor
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

personas = {}


# persona_traits = {
# 'leaving_home_and_coming_back': {"short" : [], "full_workday" : [], "never":[]}, 
# 'leave_home': {"early" : [], "late" : [], "at_night" : [], "multiple_times": [], "never":[]}, 
# 'come_back': {"early" : [], "late" : [], "at_night" : [], "multiple_times": [], "never":[]}, 
# 'playing_music': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]}, 
# 'getting_dressed': {"for_work":[], "for_evening":[], "morning_andEvening":[], "not_at_all":[]}, 
# 'cleaning': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]}, 
# 'breakfast': {"has_breakfast":[], "skips_breakfast":[]}, 
# 'socializing': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]}, 
# 'lunch': {"has_lunch":[], "skips_lunch":[]}, 
# 'going_to_the_bathroom': {"over three times":[], "under_three_times":[]}, 
# 'listening_to_music':  {"morning":[], "evening":[], "morning_and_evening":[], "not_at_all":[]}, 
# 'taking_medication': {"morning/noon":[], "evening":[], "twice":[], "never":[]}, 
# 'take_out_trash': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]}, 
# 'kitchen_cleaning': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]}, 
# 'dinner': {"early":[], "on_time":[], "late":[]}, 
# 'wash_dishes': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]},  
# 'brushing_teeth': {"morning_only":[], "twice":[]}, 
# 'laundry': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]}, 
# 'hand_wash_clothes': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]}, 
# 'reading': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]}, 
# 'showering': {"morning":[], "evening":[], "twice":[]}, 
# 'computer_work': {"work_from_home_day":[], "sparse":[]}, 
# 'vaccuum_cleaning': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]},  
# 'watching_tv': {"morning":[], "evening":[], "multiple_times":[], "not_at_all":[]}
# }


# early riser, works long hours, has less time for chores
personas['hard worker'] = {
    'leave_home' : 'early',
    'come_back' : 'late',
    'playing_music' : 'not_at_all',
    'getting_dressed' : 'for_work',
    'cleaning' : 'evening',
    'breakfast' : 'has_breakfast',
    'socializing' : 'not_at_all',
    'lunch' : 'skips_lunch',
    'going_to_the_bathroom' : 'under_three_times',
    'listening_to_music' : 'not_at_all',
    'taking_medication' : 'never',
    'take_out_trash' : 'not_at_all',
    'kitchen_cleaning' : 'not_at_all',
    'dinner' : 'late',
    'wash_dishes' : 'evening',
    'brushing_teeth' : 'twice',
    'laundry' : 'not_at_all',
    'reading' : 'evening',
    'showering' : 'twice',
    'computer_work' : 'sparse',
    'vaccuum_cleaning' : 'not_at_all',
    'watching_tv' : 'not_at_all',
}

# early riser, works from home, enjoys evenings with tv, music and friends 
personas['work from home'] = {
    'leave_home' : 'never',
    'come_back' : 'never',
    'playing_music' : 'evening',
    'getting_dressed' : 'for_evening',
    'cleaning' : 'evening',
    'breakfast' : 'has_breakfast',
    'socializing' : 'evening',
    'lunch' : 'has_lunch',
    'going_to_the_bathroom' : 'under_three_times',
    'listening_to_music' : 'evening',
    'taking_medication' : 'morning/noon',
    'take_out_trash' : 'evening',
    'kitchen_cleaning' : 'not_at_all',
    'dinner' : 'on_time',
    'wash_dishes' : 'not_at_all',
    'brushing_teeth' : 'morning_only',
    'laundry' : 'not_at_all',
    'reading' : 'evening',
    'showering' : 'evening',
    'computer_work' : 'work_from_home_day',
    'vaccuum_cleaning' : 'not_at_all',
    'watching_tv' : 'evening',
}

# home maker, less computer work, lots of chores, enjoys evenings with tv, music and friends 
personas['home maker'] = {
    'leave_home' : 'late',
    'come_back' : 'late',
    'playing_music' : 'evening',
    'getting_dressed' : 'for_evening',
    'cleaning' : 'evening',
    'breakfast' : 'has_breakfast',
    'socializing' : 'evening',
    'lunch' : 'has_lunch',
    'going_to_the_bathroom' : 'under_three_times',
    'listening_to_music' : 'evening',
    'taking_medication' : 'evening',
    'take_out_trash' : 'evening',
    'kitchen_cleaning' : 'morning',
    'dinner' : 'on_time',
    'wash_dishes' : 'morning',
    'brushing_teeth' : 'morning_only',
    'laundry' : 'morning',
    'reading' : 'evening',
    'showering' : 'twice',
    'computer_work' : 'sparse',
    'vaccuum_cleaning' : 'morning',
    'watching_tv' : 'not_at_all',
}

# elderly, less work and no going out, sparsely indulges in leisurely activities 
personas['elderly'] = {
    'leave_home' : 'never',
    'come_back' : 'never',
    'playing_music' : 'evening',
    'getting_dressed' : 'not_at_all',
    'cleaning' : 'not_at_all',
    'breakfast' : 'has_breakfast',
    'socializing' : 'not_at_all',
    'lunch' : 'has_lunch',
    'going_to_the_bathroom' : 'under_three_times',
    'listening_to_music' : 'evening',
    'taking_medication' : 'twice',
    'take_out_trash' : 'evening',
    'kitchen_cleaning' : 'not_at_all',
    'dinner' : 'early',
    'wash_dishes' : 'not_at_all',
    'brushing_teeth' : 'twice',
    'laundry' : 'not_at_all',
    'reading' : 'evening',
    'showering' : 'twice',
    'computer_work' : 'sparse',
    'vaccuum_cleaning' : 'morning',
    'watching_tv' : 'evening',
}


activity_map = {
"brush_teeth" : "brushing_teeth",
"bathe_shower" : "showering",
"prepare_eat_breakfast" : "breakfast",
"get_dressed" : "getting_dressed",
"computer_work" : "computer_work",
"prepare_eat_lunch" : "lunch",
"leave_home" : "leave_home",
"come_home" : "come_back",
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
"sleep" : None
}
start_times = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

class ScheduleDistributionSampler():
    def __init__(self, type):
        activity_histogram = {}
        if type.lower() == 'persona':
            with open('data/personaBasedSchedules/trait_histograms.json') as f:
                trait_histograms = json.load(f)
            persona_name = random.choice(list(personas.keys()))
            self.label = persona_name
            persona = personas[persona_name]
            for activity in persona:
                try:
                    activity_histogram[activity] = trait_histograms[activity][persona[activity]]
                except Exception as e:
                    print(activity, persona_name)
                    raise e
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
        self.sampling_range = max(self.activity_threshold[-1,:]) * 1.2
        self.removed_activities = []
        self.next_activity_must_be_one_of = []

    def __call__(self, t_mins, remove=False):
        sample = random.random()*self.sampling_range
        activity = None
        st = start_times.index(int(floor(t_mins/60)))
        for act, thresh in zip(self.activities, list(self.activity_threshold[:,st])):
            if thresh > sample:
                activity = act
                break
        if activity in self.removed_activities:
            return None
        if self.next_activity_must_be_one_of and activity not in self.next_activity_must_be_one_of:
            return None
        else:
            if remove: self.remove(activity)
        if activity == "leave_home":
            self.next_activity_must_be_one_of = ["come_home"]
        return activity

    def remove(self, activity):
        self.removed_activities.append(activity)

    def plot(self, filepath = None):
        # clrs = sns.color_palette("pastel") + sns.color_palette("dark") + sns.color_palette()
        fig, ax = plt.subplots()
        fig.set_size_inches(27, 18.5)
        base = self.activity_threshold[0,:] * 0
        for i, activity in enumerate(self.activities):
            self.activity_threshold[i,:]
            d = self.activity_threshold[i,:] - base
            # ax.bar(start_times, d, label=activity, bottom=base, color=clrs[i])
            ax.bar(start_times, d, label=activity, bottom=base)
            base = self.activity_threshold[i,:]
        ax.set_xticks(start_times)
        ax.set_xticklabels([str(s)+':00' for s in start_times])
        ax.set_title(self.label)
        plt.legend()
        if filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()

