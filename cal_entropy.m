function result=cal_entropy(my_data)
    temp_size=length(my_data(:,1));
    temp_sum=sum(my_data(:,1));
    p1=(temp_size+temp_sum)/2/temp_size;
    p2=(temp_size-temp_sum)/2/temp_size;
    if (p1==0) || (p2==0)
        result=0;
    else
        result=-p1*log2(p1)-p2*log2(p2);
    end
end