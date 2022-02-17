import numpy as np
from typing import Dict, List
from Interface.RewardInterface import Reward


class ScheduleReward(Reward):
    def __init__(self, jobs_info: Dict[str, float], equipments_info: List[str], actions, deadline):
        self.jobs = list(jobs_info.keys())
        self.jobs_run_time = jobs_info  # 先這樣
        self.eqps = equipments_info
        self.jobs_count = len(self.jobs)
        self.eqps_count = len(self.eqps)
        self.actions = actions
        self.deadline = deadline

    def get_episode_ended_reward(self, state, step_count) -> float:
        return self.get_remain_job_reward(state) + self.get_utilization_rate_reward(
            state) + self.get_deadline_reward(state) #+ self.get_step_disreward(step_count)#+self.get_repeat_job_reward(state)

    def get_repeat_job_reward(self, state, job: int = None) -> float:
        # repeat_job
        # np.where(self._state == i) eg. np.where([[0,1,3][0,1,2]]==1) return => ([0,1],[1,1])
        if job:
            repeat_job = [len(np.where(state == job)[0]) - 1]
        else:
            repeat_job = [len(np.where(state == i)[0]) - 1 for i in range(0, self.jobs_count)]
        repeat_job = sum([j for j in repeat_job if j > 0])
        return repeat_job * -10  # -300

    def get_remain_job_reward(self, state):
        not_finish_jobs_count = len([i for i in state.observation.values() if not i.is_done])
        return -3 * not_finish_jobs_count # -110 * len(remain_jobs)

    def get_utilization_rate_reward(self, state) -> float:
        return state.get_utilization_rate() * 10

    def get_episode_not_ended_reward(self, current_state: np.array, action):
        reward = 0
        action_index = action
        # Make sure episodes don't go on forever.
        # action [job,eqp] this job do in this epq
        assign_job = self.actions[action_index]
        job_name = assign_job[0]
        if current_state.observation[job_name].is_done:
            reward = -10
        else:
            reward = 10
        return reward

    def get_deadline_reward(self, state):
        over_deadline_time = [ob.end_time-self.deadline for ob in state.observation.values() if ob.end_time>self.deadline]
        return sum(over_deadline_time) * -5

    def get_step_disreward(self, step_count):
        return step_count * -1
