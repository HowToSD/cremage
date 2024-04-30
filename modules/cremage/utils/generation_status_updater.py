class StatusUpdater():
    def __init__(self, steps_per_batch:int, num_batches:int, status_queue):
        self.steps_per_batch = steps_per_batch
        self.total_steps = steps_per_batch * num_batches
        self.status_queue = status_queue
        self.batch_count = 0
        self.steps_completed = 0

    def status_update(self, step):
        self.steps_completed += 1  # Just count the frequency
        if self.steps_completed == self.total_steps:
            self.status_queue.put("Done")
        else:
            self.status_queue.put(f"{self.steps_completed} / {self.total_steps}")

