import { z } from "zod";
import { dedent, judger, type ExportedConversation } from "./index";
import { generateObject } from "ai";
import { readdirSync } from "fs";
import limit from "p-limit";

const judgingPrompt = dedent(`
You are analyzing customer service chat conversations to evaluate their quality and extract valuable metrics.

Provide scores between 0.00 and 1.00 (two decimal places) for each metric below:

## Quality Score (0.00 - 1.00)
Evaluate the efficiency and usefulness of the conversation:
- **High scores (0.80-1.00)**: Direct, focused exchanges that resolve the customer's need quickly
- **Medium scores (0.40-0.79)**: Some unnecessary back-and-forth, but generally productive
- **Low scores (0.00-0.39)**: Excessive redundant messages, circular discussions, or failure to address the customer's needs

Consider:
- Number of messages needed to resolve the issue
- Clarity of questions and answers
- Whether responses directly address what was asked
- Avoidance of repetition

## Correctness Score (0.00 - 1.00)
Evaluate the factual accuracy and relevance of information exchanged:
- **High scores (0.80-1.00)**: All information appears accurate and relevant
- **Medium scores (0.40-0.79)**: Mostly correct with minor inaccuracies or some off-topic remarks
- **Low scores (0.00-0.39)**: Contains incorrect information, personal remarks unrelated to the transaction, or significant irrelevant content

Consider:
- Accuracy of product/service information provided
- Relevance of all messages to the customer's inquiry
- Absence of personal commentary or off-topic discussions

## Grammar Score (0.00 - 1.00)
Evaluate the linguistic quality of the conversation in Catalan:
- **High scores (0.80-1.00)**: Proper Catalan throughout, minimal errors
- **Medium scores (0.40-0.79)**: Mostly correct Catalan with occasional mistakes
- **Low scores (0.00-0.39)**: Frequent errors, non-Catalan words, or use of other languages

**Important**: This conversation must be in Catalan. Apply heavy penalties for:
- Use of Spanish, English, or other languages
- Non-Catalan words or expressions (unless they are accepted loanwords)
- Poor spelling, punctuation, or syntax in either party's messages

Apply penalties proportionally to both parties when they make errors.

## Completeness Score (0.00 - 1.00)
Evaluate whether the customer's issue was fully resolved:
- **1.00**: Issue completely resolved, customer explicitly confirmed satisfaction
- **0.75-0.99**: Issue appears resolved but no explicit confirmation
- **0.40-0.74**: Partial resolution, some questions answered but not all
- **0.00-0.39**: Issue unresolved, conversation ended without solution

Consider:
- Whether all customer questions were answered
- If the original issue was addressed
- Whether the conversation ended naturally with resolution or abruptly
`);

export const judgerSchema = z.object({
  quality: z.object({
    reasoning: z.string().describe("The reasoning of the score."),
    score: z
      .number()
      .describe(
        "The score of the quality of the conversation. 0 is the worst and 10 is the best."
      ),
  }),
  correctness: z.object({
    reasoning: z.string().describe("The reasoning of the score."),
    score: z
      .number()
      .describe(
        "The score of the correctness of the conversation. 0 is the worst and 10 is the best."
      ),
  }),
  grammar: z.object({
    reasoning: z.string().describe("The reasoning of the score."),
    score: z
      .number()
      .describe(
        "The score of the grammar of the conversation. 0 is the worst and 10 is the best."
      ),
  }),
  completeness: z.object({
    reasoning: z.string().describe("The reasoning of the score."),
    score: z
      .number()
      .describe(
        "The score of the completeness of the conversation. 0 is the worst and 10 is the best."
      ),
  }),
});

export const judge = async (
  id: string,
  {
    conversation,
    endTime,
    goal,
    model,
    modelName,
    startTime,
  }: ExportedConversation,
  iteration = 0
): Promise<void> => {
  try {
    console.log(`Judging ${id}`);
    const { object } = await generateObject({
      schema: judgerSchema,
      model: judger,
      messages: [
        {
          role: "system",
          content: judgingPrompt,
        },
        {
          role: "user",
          content: JSON.stringify({ conversation }),
        },
      ],
    });
    console.log(`Judged ${id}`);

    Bun.write(
      `./judgements/${id}.json`,
      JSON.stringify(
        {
          id,
          goal,
          modelName,
          startTime,
          endTime,
          conversation,
          ...object,
        },
        null,
        2
      )
    );
  } catch (error) {
    if (iteration > 10) throw error;
    return judge(id, { conversation, endTime, goal, model, modelName, startTime }, iteration + 1);
  }
};

const limited = limit(10);

export const judgeAll = async () => {
  const files = readdirSync("./results");
  const promises: Promise<void>[] = [];
  for (const file of files) {
    if (!file.endsWith(".json")) continue;
    const [id] = file.split(".");
    if (!id) continue;
    const f = Bun.file(`./results/${file}`);
    const content: ExportedConversation = await f.json();
    promises.push(limited(() => judge(id, content)));
  }
  await Promise.all(promises);
};

await judgeAll();
