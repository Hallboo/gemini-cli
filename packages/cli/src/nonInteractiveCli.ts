/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Config,
  ToolCallRequestInfo,
  executeToolCall,
  ToolRegistry,
  shutdownTelemetry,
  isTelemetrySdkInitialized,
} from '@google/gemini-cli-core';
import {
  Content,
  Part,
  FunctionCall,
  GenerateContentResponse,
} from '@google/genai';

import { parseAndFormatApiError } from './ui/utils/errorParsing.js';

function getResponseText(response: GenerateContentResponse): string | null {
  if (response.candidates && response.candidates.length > 0) {
    const candidate = response.candidates[0];
    if (
      candidate.content &&
      candidate.content.parts &&
      candidate.content.parts.length > 0
    ) {
      // We are running in headless mode so we don't need to return thoughts to STDOUT.
      const thoughtPart = candidate.content.parts[0];
      if (thoughtPart?.thought) {
        return null;
      }
      return candidate.content.parts
        .filter((part) => part.text)
        .map((part) => part.text)
        .join('');
    }
  }
  return null;
}

export async function runNonInteractive(
  config: Config,
  input: string,
): Promise<void> {
  // Handle EPIPE errors when the output is piped to a command that closes early.
  process.stdout.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EPIPE') {
      // Exit gracefully if the pipe is closed.
      process.exit(0);
    }
  });

  const geminiClient = config.getGeminiClient();
  const toolRegistry: ToolRegistry = await config.getToolRegistry();

  const chat = await geminiClient.getChat();
  const abortController = new AbortController();
  let currentMessages: Content[] = [{ role: 'user', parts: [{ text: input }] }];
  const max_iterations = 200;
  let iteration = 0;

  try {
    while (iteration < max_iterations) {
      iteration++;
      const functionCalls: FunctionCall[] = [];

      // console.debug('### send message stream', currentMessages[currentMessages.length - 1]);
      const responseStream = await chat.sendMessageStream({
        message: currentMessages[0]?.parts || [], // Ensure parts are always provided
        config: {
          abortSignal: abortController.signal,
          tools: [
            { functionDeclarations: toolRegistry.getFunctionDeclarations() },
          ],
        },
      });

      let textResponse = '';
      // Check if streaming output is enabled via environment variable
      const streamOutput = process.env.GEMINI_CLI_STREAM_OUTPUT === 'true';
      
      for await (const resp of responseStream) {
        if (abortController.signal.aborted) {
          console.error('Operation cancelled.');
          return;
        }
        const textPart = getResponseText(resp);
        if (textPart) {
          if (streamOutput) {
            process.stdout.write(textPart);
          }
          textResponse += textPart;
        }
        if (resp.functionCalls) {
          functionCalls.push(...resp.functionCalls);
        }
      }

      // If not streaming, output the complete response at once
      if (!streamOutput) {
        process.stdout.write("response: " + textResponse);
      }
      if (functionCalls.length > 0) {
        const toolResponseParts: Part[] = [];
        console.debug('### function calls', functionCalls);

        for (const fc of functionCalls) {
          const callId = fc.id ?? `${fc.name}-${Date.now()}`;
          const requestInfo: ToolCallRequestInfo = {
            callId,
            name: fc.name as string,
            args: (fc.args ?? {}) as Record<string, unknown>,
            isClientInitiated: false,
          };

          const toolResponse = await executeToolCall(
            config,
            requestInfo,
            toolRegistry,
            abortController.signal,
          );

          if (toolResponse.error) {
            console.error(
              `Error executing tool ${fc.name}: ${toolResponse.resultDisplay || toolResponse.error.message}`,
            );
            process.exit(1);
          }

          if (toolResponse.responseParts) {
            const parts = Array.isArray(toolResponse.responseParts)
              ? toolResponse.responseParts
              : [toolResponse.responseParts];
            for (const part of parts) {
              if (typeof part === 'string') {
                toolResponseParts.push({ text: part });
              } else if (part) {
                toolResponseParts.push(part);
              }
            }
          }
        }
        currentMessages = [{ role: 'user', parts: toolResponseParts }];
        console.debug('### tool response', toolResponseParts);
      } else {
        process.stdout.write('\n'); // Ensure a final newline
        console.debug('### end of iteration:', iteration);

        // check textResponse contain Finish
        if (textResponse.includes('<Finished></Finished>')) {
          console.debug('### finished:', textResponse);
          break;
        }
        currentMessages = [{ role: 'user', parts: [
            { text: 'Please continue. if you need to call tools, please do so. if you finished the job, please answer <Finished></Finished>' }] }
        ];
      }
    }
  } catch (error) {
    console.error(
      parseAndFormatApiError(
        error,
        config.getContentGeneratorConfig().authType,
      ),
    );
    process.exit(1);
  } finally {
    // 只有在未明确禁用且telemetry已初始化的情况下才执行shutdown操作
    const disableTelemetryShutdown = process.env.GEMINI_CLI_DISABLE_TELEMETRY_SHUTDOWN === 'true';
    if (!disableTelemetryShutdown && isTelemetrySdkInitialized()) {
      try {
        await shutdownTelemetry();
      } catch (error) {
        // 忽略telemetry关闭时的错误，避免影响主程序退出
        console.debug('Error during telemetry shutdown:', error);
      }
    }
  }
}
