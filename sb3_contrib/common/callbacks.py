from stable_baselines3.common.callbacks import BaseCallback, EventCallback

class EveryEpisodeEnd(EventCallback):
    """
    Trigger a callback every time an episode ends

    :param callback: Callback that will be called
        when the event is triggered.
    """

    def __init__(self, callback: BaseCallback):
        super().__init__(callback)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]

        for info in infos:
            if info["episode_ended"]:
                return self._on_event()

        return True