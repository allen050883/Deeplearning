import numpy as np


class Jobs:
    def __init__(self, job_run_time):
        self.jobs_name = list(job_run_time.keys())
        self.job_run_time = job_run_time
        self.observation = {job_name: JobObservation(run_time) for job_name, run_time in self.job_run_time.items()}
        self.eqp_count = 2

    def get_observation_spec(self):
        return np.array([n.get_one_observation() for n in self.observation.values()]).shape

    def get_observation(self):
        return np.array([n.get_one_observation() for n in self.observation.values()])

    def get_eqp_end_time(self, eqp_id):
        eqp_state = [0 for _ in range(self.eqp_count)]
        eqp_state[eqp_id] = 1
        eqp_end_time = max([i.end_time for i in self.observation.values() if list(i.running_eqp) == eqp_state])
        return eqp_end_time

    def get_utilization_rate(self):
        eqp_pair = []
        for ob in self.observation.values():
            if list(ob.running_eqp) not in eqp_pair:
                eqp_pair.append(list(ob.running_eqp))

        eqp_run_times = [sum([ob.run_time for ob in self.observation.values()
                              if ob.is_done and pair == list(ob.running_eqp)])
                         for pair in eqp_pair]
        return sum([total_run_time / max(eqp_run_times) for total_run_time in eqp_run_times]) / self.eqp_count


class JobObservation:
    def __init__(self, run_time):
        self.eqp_count = 2
        self.is_done: float = 0.0
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.run_time:float = float(run_time)
        self.can_run_eqp = np.ones((self.eqp_count,))
        self.running_eqp = np.zeros((self.eqp_count,))

    def get_one_observation(self):
        observation = [self.is_done,
                       self.start_time,
                       self.end_time,
                       self.run_time] + list(self.can_run_eqp) + list(self.running_eqp)
        return np.array(observation,dtype=float)
