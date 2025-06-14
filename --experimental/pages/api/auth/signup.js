import { connectDB } from "@/util/database";
import bcrypt from 'bcrypt'

export default async function handler(rq, rp) {
    if (rq.method == 'POST') {
        let hash = await bcrypt.hash(rq.body.password, 10)
        // console.log(hash)
        // console.log(rq.body)
        rq.body.password = hash




        let db = (await connectDB).db('forum');
        await db.collection('user_cred').insertOne(rq.body);
        rp.status(200).json('가입성공')
    }
}
