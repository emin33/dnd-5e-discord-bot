"""Shared error-handling base classes for the bot's Views and Modals.

Audit P1 (error handling): no view or modal overrode on_error, so an
exception in a button/select/modal callback fell through to py-cord's
default handler — the user stared at a hung "thinking..." interaction and
the traceback never reached our logs. SafeView/SafeModal log the failure
with the full traceback, tell the clicking user something went wrong
(ephemeral, respecting whether the interaction was already acknowledged),
and give views with one-shot claim flags a recovery hook so a failed
callback doesn't leave the view bricked.
"""

import discord
import structlog

logger = structlog.get_logger()

_ERROR_NOTICE = "Something went wrong while handling that. Please try again."


async def _notify_user(interaction: discord.Interaction) -> None:
    """Best-effort ephemeral notice, respecting prior acknowledgement."""
    try:
        if interaction.response.is_done():
            await interaction.followup.send(_ERROR_NOTICE, ephemeral=True)
        else:
            await interaction.response.send_message(_ERROR_NOTICE, ephemeral=True)
    except Exception:
        # Notification is best-effort — the log line is the record.
        pass


class SafeView(discord.ui.View):
    """View base whose on_error logs, recovers, and notifies the user.

    py-cord awaits on_error inside the dispatch's ``except`` block, so the
    original exception context is live for ``exc_info=True``.
    """

    async def on_error(
        self,
        error: Exception,
        item: discord.ui.Item,
        interaction: discord.Interaction,
    ) -> None:
        logger.error(
            "view_callback_failed",
            view=type(self).__name__,
            item=getattr(item, "label", None) or getattr(item, "custom_id", None),
            error=str(error),
            exc_info=True,
        )
        try:
            await self.on_error_recover(error, item, interaction)
        except Exception as recovery_error:
            logger.error(
                "view_error_recovery_failed",
                view=type(self).__name__,
                error=str(recovery_error),
                exc_info=True,
            )
        await _notify_user(interaction)

    async def on_error_recover(
        self,
        error: Exception,
        item: discord.ui.Item,
        interaction: discord.Interaction,
    ) -> None:
        """Hook for views with one-shot claim flags (e.g. combat's _acted):
        release the claim / restore a usable view so a failed callback
        doesn't brick it. Default is a no-op — most views hold no claim.
        """


class SafeModal(discord.ui.Modal):
    """Modal counterpart of SafeView (py-cord modal on_error has no item)."""

    async def on_error(
        self, error: Exception, interaction: discord.Interaction
    ) -> None:
        logger.error(
            "modal_callback_failed",
            modal=type(self).__name__,
            error=str(error),
            exc_info=True,
        )
        await _notify_user(interaction)
