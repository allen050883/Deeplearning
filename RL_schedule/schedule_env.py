from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from itertools import product
from ScheduleReward import ScheduleReward
from model.job import Jobs


class ScheduleEnv(py_environment.PyEnvironment):
    # network output shape = (_observation_spec.shape[0],action count)
    def __init__(self):
        self.jobs = ['J0', 'J1', 'J2', 'J3']
        self.jobs_run_time = {'J0': 10, 'J1': 5, 'J2': 20, 'J3': 100}
        # job_info : [is_done,start_time,end_time,run_time,can_run_eqp1,can_run_eqp2,runing_eqp1,runing_eqp2]
        self.job_info = Jobs(self.jobs_run_time)
        self.obervation_spec = self.job_info.get_observation_spec()
        self.equipments = ['E0', 'E1']
        self.deadline = 100
        # action, array for jobs*equipments
        self.ac = np.array(list(product(self.jobs, self.equipments)))
        self.schedule_reward = ScheduleReward(self.jobs_run_time, self.equipments, self.ac, self.deadline)
        self.eqps_count = len(self.equipments)
        self.jobs_count = len(self.jobs)
        max_run_time = max(self.jobs_run_time.values())
        max_time = max_run_time * self.jobs_count

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(self.ac) - 1, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.obervation_spec, dtype=np.float32, minimum=[0, 0, 0, 0, 0, 0, 0, 0],
            maximum=[1, max_time, max_time, max_run_time, 1, 1, 1, 1],
            name='observation')
        # self._state = np.full((self.eqps_count, self.jobs_count), -1)
        self._state = self.job_info.get_observation()
        self._episode_ended = False
        self.max_step = 20
        self.step_count = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self.job_info.get_observation()
        self._episode_ended = False
        self.step_count = 0
        self.assigned_job = []
        return ts.restart(np.array(self._state, dtype=np.float32))

    def update_state(self, eqp_id, job_name):
        if not self.job_info.observation[job_name].is_done:
            self.job_info.observation[job_name].running_eqp[eqp_id] = 1
            start_time = self.job_info.get_eqp_end_time(eqp_id)
            end_time = start_time + self.jobs_run_time[job_name]
            self.job_info.observation[job_name].is_done = 1
            self.job_info.observation[job_name].start_time = start_time
            self.job_info.observation[job_name].end_time = end_time
        self._state = self.job_info.get_observation()

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        self.step_count += 1
        action_index = action
        # Make sure episodes don't go on forever.
        # action [job,eqp] this job do in this epq
        assign_job = self.ac[action_index]
        eqp_name = assign_job[1]
        job_name = assign_job[0]
        eqp_index = self.equipments.index(eqp_name)
        job_index = self.jobs.index(job_name)
        is_all_job_assigned = all([v.is_done for v in self.job_info.observation.values()])  # all job is done
        if is_all_job_assigned or self.step_count == self.max_step:
            self._episode_ended = True

        # Agent take infinite step to take action, refine it in 100 steps
        if self._episode_ended:
            self.update_state(eqp_index, job_name)
            reward = self.schedule_reward.get_episode_ended_reward(self.job_info, self.step_count)
            # print("[INFO] state => \n", self._state)
            # print("[INFO] reward => ", reward)
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            reward = self.schedule_reward.get_episode_not_ended_reward(self.job_info, action)
            self.update_state(eqp_index, job_name)
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=0.7)
